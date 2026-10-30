"""Evaluate the reasoning agent on the public datasets.

Usage:
    uv run python experiments/evaluate_all.py [--ruletaker-limit N] [--split dev]
"""

import argparse

from reasoning_agent.agent import ReasoningAgent
from reasoning_agent.data import load_counterbench, load_folio, load_ruletaker
from reasoning_agent.evaluate import breakdown, run, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruletaker-limit", type=int, default=0, help="0 = whole split")
    parser.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    parser.add_argument("--strategy", default="forward", choices=("forward", "backward"))
    parser.add_argument("--no-fol", action="store_true", help="ignore FOLIO FOL annotations")
    parser.add_argument("--only", choices=("ruletaker", "folio", "counterbench"))
    args = parser.parse_args()

    agent = ReasoningAgent(strategy=args.strategy)
    datasets = {
        "ruletaker": lambda: load_ruletaker(args.split, args.ruletaker_limit or None),
        "folio": lambda: load_folio("train") + load_folio("validation"),
        "counterbench": load_counterbench,
    }
    for name, loader in datasets.items():
        if args.only and name != args.only:
            continue
        examples = loader()
        results = run(agent, examples, use_fol=not args.no_fol)
        text, _, _ = summarize(results, f"{name} ({len(examples)} examples)")
        print(text)
        if name == "ruletaker":
            for k, (c, n) in breakdown(results, lambda e: e["meta"]["config"]).items():
                print(f"    {k:>20}: {c}/{n} = {100 * c / n:.2f}%")
        if name == "counterbench":
            for k, (c, n) in breakdown(results, lambda e: e["meta"]["type"]).items():
                print(f"    {k:>20}: {c}/{n} = {100 * c / n:.2f}%")
        if name == "folio":
            for k, (c, n) in breakdown(results, lambda e: e["meta"]["raw_label"]).items():
                print(f"    gold {k:>15}: {c}/{n} = {100 * c / n:.2f}%")


if __name__ == "__main__":
    main()
