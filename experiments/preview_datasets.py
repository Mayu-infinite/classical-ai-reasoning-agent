
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def show_example(dataset_name, split, limit=2):
    print(f"\n{'=' * 70}")
    print(f"{dataset_name} | {split}")
    print("=" * 70)

    try:
        dataset = load_dataset(dataset_name, split=split)
        print("Number of examples:", len(dataset))
        print("Columns:", dataset.column_names)

        for index in range(min(limit, len(dataset))):
            print(f"\nExample {index + 1}:")
            print(json.dumps(dataset[index], indent=2, ensure_ascii=False, default=str))

    except Exception as exc:
        print(f"Could not load split: {exc}")


def show_counterbench(filename):
    print(f"\n{'=' * 70}")
    print(f"CounterBench | {filename}")
    print("=" * 70)

    try:
        path = hf_hub_download(
            repo_id="CounterBench/CounterBench",
            filename=filename,
            repo_type="dataset",
        )

        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            print("Number of records:", len(data))
            examples = data[:2]
        elif isinstance(data, dict):
            print("Top-level keys:", list(data.keys()))
            examples = {"preview": data}
        else:
            print("Unexpected data type:", type(data))
            return

        print(json.dumps(examples, indent=2, ensure_ascii=False, default=str)[:12000])

    except Exception as exc:
        print(f"Could not inspect file: {exc}")


def main():
    show_example("tasksource/ruletaker", "train")
    show_example("tasksource/folio", "train")

    show_counterbench("data_balanced_alpha_V1.json")
    show_counterbench("data_balanced_backdoor_V2.json")


if __name__ == "__main__":
    main()
