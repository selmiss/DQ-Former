
import argparse
from tqdm import tqdm
import re
import numpy as np
import os

from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from transformers import AutoTokenizer, LlamaForCausalLM
from models.mol_llama import MolLLaMA, DQMolLLaMA, DQMolLLaMAEncoder
from utils.configuration_mol_llama import MolLLaMAConfig

from dataset import ZeroshotDataset, ZeroshotCollater


def main(args):
    # Load model and tokenizer
    llama_version = 'llama3' if 'Llama-3' in args.pretrained_model_name_or_path else 'llama2'
    if args.tokenizer_path is None:
        tokenizer_path = args.pretrained_model_name_or_path
    else:
        tokenizer_path = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]
    tokenizer.padding_side = 'left'
    if llama_version == 'llama3':
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    elif llama_version == 'llama2':
        terminators = tokenizer.eos_token_id
    
    # Initialize model directly instead of using from_pretrained
    config = MolLLaMAConfig(
        llm_config={'llm_model': args.pretrained_model_name_or_path},
        qformer_config={'use_dq_encoder': args.use_dq_encoder, 'use_flash_attention': True},  # Adjust as needed
        graph_encoder_config={'encoder_types': ['unimol']},  # Adjust as needed
        torch_dtype="float16"
    )
    if args.use_dq_encoder:
        model = DQMolLLaMA(
            config=config,
            vocab_size=len(tokenizer),
            torch_dtype="float16",
            enable_flash=True
        )
        model.load_from_ckpt(args.qformer_path)
        encoder = model.encoder
    elif args.only_llm:
        model = LlamaForCausalLM.from_pretrained(args.pretrained_model_name_or_path)
        encoder = DQMolLLaMAEncoder(
            graph_encoder_config = config.graph_encoder_config,
            blending_module_config = config.blending_module_config,
            qformer_config = config.qformer_config,
        )
    else:
        model = MolLLaMA(
            config=config,
            vocab_size=len(tokenizer),
            torch_dtype="float16",
            enable_flash=True
        )
        model.load_from_ckpt(args.qformer_path)
        encoder = model.encoder


    model = model.to(args.device)

    data_dir = os.path.join(args.data_dir, 'zeroshot', args.task_name)

    dataset = ZeroshotDataset(data_dir=data_dir, 
                        split='test', prompt_type=args.prompt_type, 
                        unimol_dictionary=encoder.unimol_dictionary,
                        only_llm=args.only_llm)

    collater = ZeroshotCollater(tokenizer, encoder.unimol_dictionary, llama_version, args.only_llm)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collater, shuffle=False)

    pattern = r"[Ff]inal [Aa]nswer:"

    responses, answers, smiles_list = [], [], []
    for graph_batch, text_batch, answer, smiles in tqdm(dataloader):
        for key in graph_batch.keys():
            if key == 'unimol':
                for key_ in graph_batch[key].keys():
                    graph_batch[key][key_] = graph_batch[key][key_].to(args.device)
            elif key == 'moleculestm':
                graph_batch[key] = graph_batch[key].to(args.device)
        text_batch = text_batch.to(args.device)

        # Generate
        if args.only_llm:
            outputs = model.generate(
                inputs = text_batch['input_ids'],
                attention_mask = text_batch['attention_mask'],
                max_new_tokens = 1024,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = terminators,
            )
        else:
            outputs = model.generate(
                graph_batch = graph_batch,
                text_batch = text_batch,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = terminators,
            )
        
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        original_texts = tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)

        # Generate further if the output does not contain "Final answer:"
        no_format_indices = []
        new_texts = []
        for idx, (original_text, generated_text) in enumerate(zip(original_texts, generated_texts)):
            if not re.search(pattern, generated_text):
                no_format_indices.append(idx)
                new_texts.append(original_text + generated_text + "\n\nFinal answer: ")
        
        if len(no_format_indices) > 0:
            new_graph_batch = {"unimol": {}, "moleculestm": {}}
            new_text_batch = {}
            for k, v in graph_batch['unimol'].items():
                new_graph_batch['unimol'][k] = v[no_format_indices]
            new_graph_batch['moleculestm'] = Batch.from_data_list(graph_batch['moleculestm'].index_select(no_format_indices))

            new_text_batch = tokenizer(
                new_texts,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to(args.device)
            new_text_batch.mol_token_flag = (new_text_batch.input_ids == tokenizer.mol_token_id).to(args.device)

            new_generated_texts = model.generate(
                graph_batch = new_graph_batch,
                text_batch = new_text_batch,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = terminators,
            )

            new_generated_texts = tokenizer.batch_decode(new_generated_texts, skip_special_tokens=True)

            for _, i in enumerate(no_format_indices):
                generated_texts[i] += "\n\nFinal answer: " + new_generated_texts[_]
            
        responses.extend(generated_texts)
        answers.extend(answer)
        smiles_list.extend(smiles)

    
    true_pattern = r'[Hh]igh [Pp]ermeability'
    false_pattern = r'[Ll]ow-to-[Mm]oderate [Pp]ermeability|[Mm]oderate [Pp]ermeability'
    labels, preds = [], []
    for response, answer in zip(responses, answers):
        label = 1 if answer == "High permeability" else 0

        response = response.split("Final answer: ")[-1].strip()
        if re.search(true_pattern, response): pred = 1
        elif re.search(false_pattern, response): pred = 0
        else: pred = None

        labels.append(label)
        preds.append(pred)


    # Save the results
    save_name = f"{'dq' if args.use_dq_encoder else 'no_dq'}_{'only_lm' if args.only_llm else 'no_only_lm'}_{llama_version}_{args.prompt_type}"
    output_dir = os.path.join(args.data_dir, 'results', args.task_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f're_{save_name}.txt'), 'w', encoding='utf-8') as f:
        for response, answer, smiles, label, pred in zip(responses, answers, smiles_list, labels, preds):
            f.write(f"SMILES: {smiles}\n")
            f.write('-'*50 + "\n")
            f.write(f"Label: {label}\n")
            f.write(f"Prediction: {pred if pred is not None else 'None'}\n")
            f.write('-'*50 + "\n")
            f.write(f"Response: {response}\n")
            f.write('-'*50 + "\n")
            f.write(f"Answer: {answer}\n")
            f.write("="*50 + "\n")

    # Calculate accuracy
    preds, labels = np.array(preds), np.array(labels)
    mask = preds != None
    labels = labels[mask]
    preds = preds[mask]


    accuracy = (preds == labels).sum() / len(labels) * 100
    print(f'Accuracy: {accuracy:.2f}%')

    with open(os.path.join(output_dir, f'acc_{save_name}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {accuracy:.2f}%\n')            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, default='pampa')
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--qformer_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--prompt_type', type=str, default='default', choices=['default', 'rationale', 'task_info'],)
    parser.add_argument('--only_llm', default=False, action='store_true')
    parser.add_argument('--use_dq_encoder', default=True, action='store_true')
    args = parser.parse_args()
    main(args)

