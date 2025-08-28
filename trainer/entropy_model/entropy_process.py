from trainer.entropy_model.entropy import score_batch
import torch
import json, argparse
from pathlib import Path
from tqdm import tqdm

def group_ids_by_entropy(smiles, ckpt_dir, vocab_path, q=0.75):
    
    ent_list, _ = score_batch(
        smiles_batch=smiles,
        ckpt_dir=ckpt_dir,
        vocab_path=vocab_path
    )
    groups, thresholds = [], []
    for e in ent_list:                      # e: 1D tensor of entropies per step (incl. <eos>)
        e = e.detach().cpu().float()
        if e.numel() == 0:
            groups.append([]); thresholds.append(float("nan")); continue

        e_atoms = e[1:] if e.numel() > 1 else e   # drop <bos> and <eos>
        thr = torch.quantile(e_atoms, q)
        thresholds.append(thr.item())

        gid, prev_high, ids = 0, False, []
        for v in e_atoms:                   # one id per atom (original order)
            high = v > thr
            ids.append(int(gid))
            if high and not prev_high:      # rising edge => new subgroup
                gid += 1
            prev_high = bool(high)
        groups.append(ids)
    return groups, thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json",  required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--vocab",    required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--q",          type=float, default=0.75)
    args = ap.parse_args()

    # Detect format based on file extension
    is_jsonl = args.in_json.lower().endswith('.jsonl')
    
    # Read input file
    if is_jsonl:
        # Parse JSONL format (one JSON object per line)
        data = []
        with open(args.in_json, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
    else:
        # Parse JSON format
        try:
            data = json.loads(Path(args.in_json).read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    N = len(data)
    iters = (N + args.batch_size - 1) // args.batch_size

    for s in tqdm(range(0, N, args.batch_size), total=iters, ncols=0, desc="Processing"):
        batch = data[s : s + args.batch_size]
        smiles = [d["smiles"] for d in batch]
        gids_list, _ = group_ids_by_entropy(smiles, args.ckpt_dir, args.vocab, q=args.q)
        for rec, gids in zip(batch, gids_list):
            rec["entropy_gids"] = gids

    # Write output in the same format as input
    if is_jsonl:
        # Write as JSONL (one JSON object per line)
        with open(args.out_json, "w", encoding="utf-8") as f:
            for record in data:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
    else:
        # Write as JSON
        Path(args.out_json).write_text(json.dumps(data, ensure_ascii=False))
    
    print(f"Saved -> {args.out_json}")

if __name__ == "__main__":
    main()



