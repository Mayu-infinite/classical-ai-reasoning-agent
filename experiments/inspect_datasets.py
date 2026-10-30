
from datasets import get_dataset_config_names, get_dataset_split_names
from huggingface_hub import HfApi


def inspect_dataset(repo_id: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Dataset: {repo_id}")

    try:
        configs = get_dataset_config_names(repo_id)
        print("Configurations:", configs)

        for config in configs:
            try:
                splits = get_dataset_split_names(
                    repo_id,
                    config_name=config,
                )
                print(f"  {config}: {splits}")
            except Exception as exc:
                print(f"  Could not inspect {config}: {exc}")

    except Exception as exc:
        print(f"Inspection failed: {exc}")


def main() -> None:
    inspect_dataset("tasksource/ruletaker")
    inspect_dataset("tasksource/folio")

    print(f"\n{'=' * 60}")
    print("Dataset: CounterBench/CounterBench")

    try:
        api = HfApi()
        files = api.list_repo_files(
            "CounterBench/CounterBench",
            repo_type="dataset",
        )

        for filename in files:
            if filename.endswith((".json", ".jsonl", ".csv", ".parquet")):
                print(filename)

    except Exception as exc:
        print(f"Inspection failed: {exc}")


if __name__ == "__main__":
    main()
