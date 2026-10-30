
import argparse
import time
from collections import Counter

from datasets import load_dataset

from reasoning_agent.data.ruletaker import adapt_ruletaker_record
from reasoning_agent.inference.engine import ReasoningEngine
from reasoning_agent.parsing.rule_parser import (
    parse_atom,
    parse_knowledge_base,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the supported RuleTaker subset."
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--strategy",
        choices=("forward", "backward"),
        default="forward",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be greater than zero")

    print("Loading the public RuleTaker development split...")
    dataset = load_dataset("tasksource/ruletaker", split="dev")

    sample_size = min(args.limit, len(dataset))
    sample = dataset.shuffle(seed=args.seed).select(range(sample_size))

    engine = ReasoningEngine()
    correct = 0
    supported = 0
    unsupported_reasons = Counter()
    total_inference_time = 0.0

    for record in sample:
        adapted, reason = adapt_ruletaker_record(dict(record))

        if adapted is None:
            unsupported_reasons[reason or "Unknown reason"] += 1
            continue

        try:
            knowledge_base = parse_knowledge_base(
                adapted["facts"],
                adapted["rules"],
            )
            query = parse_atom(adapted["query"])

            start = time.perf_counter()
            prediction = engine.entails(
                knowledge_base,
                query,
                strategy=args.strategy,
            )
            elapsed = time.perf_counter() - start

        except (ValueError, TypeError, RuntimeError) as exc:
            unsupported_reasons[
                f"Reasoning/parser error: {type(exc).__name__}: {exc}"
            ] += 1
            continue

        total_inference_time += elapsed
        supported += 1
        correct += int(prediction == adapted["label"])

    unsupported = sample_size - supported
    coverage = supported / sample_size
    accuracy = correct / supported if supported else 0.0

    print("\n" + "=" * 60)
    print("RULETAKER PUBLIC DEVELOPMENT EVALUATION")
    print("=" * 60)
    print("Split: dev")
    print(f"Random seed: {args.seed}")
    print(f"Strategy: {args.strategy}")
    print(f"Sample size: {sample_size}")
    print(f"Supported and evaluated: {supported}")
    print(f"Unsupported: {unsupported}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Correct among supported: {correct}")
    print(f"Accuracy among supported: {accuracy:.2%}")
    print(f"Total inference time: {total_inference_time:.6f} seconds")

    if supported:
        print(
            "Average inference time: "
            f"{total_inference_time / supported:.8f} seconds"
        )

    if unsupported_reasons:
        print("\nMost common unsupported reasons:")
        for reason, count in unsupported_reasons.most_common(10):
            print(f"  {count:4d}  {reason}")

    print("\nInterpretation:")
    print(
        "Accuracy is conditional on examples accepted by the limited "
        "adapter. It is not full RuleTaker benchmark accuracy."
    )
    print(
        "The engine uses a restricted positive-Horn representation; "
        "this evaluation does not establish full negation semantics."
    )


if __name__ == "__main__":
    main()
