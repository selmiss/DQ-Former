import json
import os
from typing import Dict


def get_project_root() -> str:
    """Return absolute path to project root (one level above this file's directory)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_new_prompts() -> Dict[str, Dict[str, str]]:
    """
    Build a dictionary of additional prompt templates to be merged into each
    zeroshot meta.json file.

    The keys (prompt types) are shared across all datasets. Prompts are written
    to be task-agnostic but still focused on binary molecular property
    prediction with the same output format as existing prompts.
    """
    system_base_direct = (
        "You are an expert molecular property prediction assistant for a binary task. "
        "For each molecule, you must decide whether it belongs to the positive (Active) "
        "or negative (Inactive) class defined by this dataset. "
        "Always follow the required answer format exactly.\n"
        "Your final answer should be formatted as either: 'Final answer: Active' or "
        "'Final answer: Inactive'"
    )

    system_base_rationale = (
        "You are an expert molecular property prediction assistant. "
        "You should briefly reason about structural and physicochemical features "
        "of the molecule before deciding whether it is Active or Inactive. "
        "However, your final line must obey the required answer format.\n"
        "Your final answer should be formatted as either: 'Final answer: Active' or "
        "'Final answer: Inactive'"
    )

    system_base_task = (
        "You are a molecular property prediction assistant working on a specific "
        "binary dataset task (e.g., activity, toxicity, or ADME classification). "
        "Use your knowledge of medicinal chemistry, ADMET principles, and structure–"
        "property relationships to map molecules to the Active or Inactive class. "
        "Do not change the required answer format.\n"
        "Your final answer should be formatted as either: 'Final answer: Active' or "
        "'Final answer: Inactive'"
    )

    system_binary = (
        "You are a strict binary classifier for molecular property prediction. "
        "You are only allowed to output one of two labels, using the required format.\n"
        "Your final answer must be exactly one of: 'Final answer: Active' or "
        "'Final answer: Inactive'"
    )

    system_confidence = (
        "You are a careful molecular property prediction assistant. "
        "First, internally estimate how confident you are that the molecule is Active "
        "or Inactive, but do NOT include numerical probabilities in the output. "
        "After reasoning, output only the final decision using the required format.\n"
        "Your final answer should be formatted as either: 'Final answer: Active' or "
        "'Final answer: Inactive'"
    )

    system_checklist = (
        "You are a molecular property prediction assistant that follows a short checklist "
        "before answering: (1) identify key substructures; (2) consider lipophilicity, "
        "polarity, and size; (3) consider potential functional-group alerts; then (4) "
        "decide whether the molecule is Active or Inactive. "
        "Despite using this checklist, you must keep the final answer format unchanged.\n"
        "Your final answer should be formatted as either: 'Final answer: Active' or "
        "'Final answer: Inactive'"
    )

    return {
        "default_variant_1": {
            "system": system_base_direct,
            "user": (
                "Classify the following molecule as Active or Inactive according to this "
                "dataset's task.\nMolecule <mol>."
            ),
        },
        "default_variant_2": {
            "system": system_base_direct,
            "user": (
                "Based on its structure, decide whether the molecule should be labeled "
                "Active or Inactive for this task. Provide only the classification in the "
                "required final answer format.\nMolecule <mol>."
            ),
        },
        "default_variant_3": {
            "system": system_base_direct,
            "user": (
                "For the molecule below, determine the correct binary label (Active or "
                "Inactive) for this dataset and follow the exact final answer format.\n"
                "Molecule <mol>."
            ),
        },
        "rationale_variant_1": {
            "system": system_base_rationale,
            "user": (
                "Briefly explain the main structural or physicochemical reasons for your "
                "decision, then give the final Active or Inactive label in the required "
                "format.\nMolecule <mol>."
            ),
        },
        "rationale_variant_2": {
            "system": system_base_rationale,
            "user": (
                "Consider potential substructures, electronic features, and global "
                "properties of the molecule, summarize your reasoning in one or two "
                "sentences, and then output the final Active or Inactive label using the "
                "required final answer format.\nMolecule <mol>."
            ),
        },
        "task_info_variant_1": {
            "system": system_base_task,
            "user": (
                "Using your understanding of typical structure–property relationships in "
                "drug discovery, decide whether the molecule should be classified as "
                "Active or Inactive in this dataset.\nMolecule <mol>."
            ),
        },
        "task_info_variant_2": {
            "system": system_base_task,
            "user": (
                "Treat this as a realistic drug discovery task. Infer whether the molecule "
                "belongs to the positive (Active) or negative (Inactive) class defined by "
                "the dataset, and output only the final label using the required format.\n"
                "Molecule <mol>."
            ),
        },
        "binary_instruction": {
            "system": system_binary,
            "user": (
                "Decide whether the molecule is Active or Inactive for this dataset. "
                "Do not output any explanation, only the final answer line.\n"
                "Molecule <mol>."
            ),
        },
        "confidence_instruction": {
            "system": system_confidence,
            "user": (
                "Carefully consider ambiguous or borderline cases internally, but in your "
                "visible output provide only the final Active or Inactive label using the "
                "required final answer format.\nMolecule <mol>."
            ),
        },
        "checklist_instruction": {
            "system": system_checklist,
            "user": (
                "Follow your internal checklist to analyze the molecule and then decide "
                "whether it should be labeled Active or Inactive. In the visible output, "
                "you may include a brief rationale, but the last line must follow the "
                "required final answer format.\nMolecule <mol>."
            ),
        },
    }


def main() -> None:
    root = get_project_root()
    zeroshot_dir = os.path.join(root, "data", "zeroshot")
    new_prompts = build_new_prompts()

    if not os.path.isdir(zeroshot_dir):
        raise SystemExit(f"Zeroshot directory not found: {zeroshot_dir}")

    # Process all dataset subdirectories under data/zeroshot
    datasets = sorted(
        d
        for d in os.listdir(zeroshot_dir)
        if os.path.isdir(os.path.join(zeroshot_dir, d))
    )

    print(f"Found zeroshot datasets: {datasets}")

    for dataset in datasets:
        meta_path = os.path.join(zeroshot_dir, dataset, "meta.json")
        if not os.path.isfile(meta_path):
            print(f"[skip] meta.json not found for dataset '{dataset}' at {meta_path}")
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        prompts = meta.get("prompts", {})
        before = len(prompts)

        # Merge new prompts, without overwriting any existing keys
        for key, value in new_prompts.items():
            if key not in prompts:
                prompts[key] = value

        after = len(prompts)
        meta["prompts"] = prompts

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"[updated] {dataset}: prompts {before} -> {after} "
            f"(added {after - before} new prompt types)"
        )


if __name__ == "__main__":
    main()


