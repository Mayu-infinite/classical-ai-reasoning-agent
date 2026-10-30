"""Run a local example through the Horn or propositional reasoning path."""
import argparse
import json

from reasoning_agent.logic.propositional import entails as prop_entails
from reasoning_agent.inference.engine import ReasoningEngine
from reasoning_agent.parsing.rule_parser import parse_atom, parse_knowledge_base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("horn", "propositional"), required=True)
    parser.add_argument("--input", required=True, help="Path to a JSON input file")
    parser.add_argument("--strategy", choices=("forward", "backward"), default="forward")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as stream:
        data = json.load(stream)
    if args.mode == "propositional":
        answer = prop_entails(data["premises"], data["query"])
    else:
        kb = parse_knowledge_base(data.get("facts", []), data.get("rules", []))
        answer = ReasoningEngine().entails(kb, parse_atom(data["query"]), strategy=args.strategy)
    print(json.dumps({"entailed": answer, "mode": args.mode, "strategy": args.strategy}, indent=2))


if __name__ == "__main__":
    main()
