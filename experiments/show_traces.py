"""Print reasoning traces (proof trees / refutations / causal worlds) for the report.

Usage:
    uv run python experiments/show_traces.py [--n 2]
"""

import argparse

from reasoning_agent.agent import ReasoningAgent, explain_text
from reasoning_agent.data import load_counterbench, load_folio, load_ruletaker


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2, help="examples per category")
    args = parser.parse_args()
    agent = ReasoningAgent()

    section("Assignment example (forward chaining)")
    kb = ("Bob is young. Bob is kind. If someone is young, then they are smart. "
          "If someone is smart and kind, then they are successful.")
    print(explain_text(agent, kb, "Bob is successful."))
    print()
    print(explain_text(ReasoningAgent(strategy="backward"), kb, "Bob is successful."))

    ruletaker = load_ruletaker("test")
    section("RuleTaker: entailed queries with multi-step proofs")
    shown = 0
    for e in ruletaker:
        if e["label"] and e["meta"]["config"] in ("depth-3", "depth-5"):
            text = explain_text(agent, e["context"], e["query"], max_context=0)
            if text.count("[rule:") >= 3:
                print(text, f"\nGold: {e['label']}\n")
                shown += 1
                if shown == args.n:
                    break

    section("RuleTaker: queries that are not entailed (closed-world assumption)")
    for e in [e for e in ruletaker if not e["label"] and e["meta"]["config"] == "depth-2"][:args.n]:
        text = explain_text(agent, e["context"], e["query"], max_context=0)
        print(text, f"\nGold: {e['label']}\n")

    section("FOLIO: resolution refutations (premises-FOL + conclusion-FOL)")
    shown = 0
    for e in load_folio("validation"):
        pred = agent.predict(e["context"], e["query"], e["premises_fol"], e["conclusion_fol"])
        if pred.label and len(pred.trace) <= 20:
            print("Premises (NL):", e["context"])
            print("Conclusion (NL):", e["query"])
            print("Premises (FOL):\n" + e["premises_fol"])
            print("Conclusion (FOL):", e["conclusion_fol"])
            print(f"Prediction: {pred.label} | gold: {e['meta']['raw_label']}")
            print("\n".join(pred.trace), "\n")
            shown += 1
            if shown == args.n:
                break

    section("FOLIO: conclusion not proved (search saturates or hits the time budget)")
    for e in [e for e in load_folio("validation") if e["meta"]["raw_label"] == "Uncertain"][:1]:
        pred = agent.predict(e["context"], e["query"], e["premises_fol"], e["conclusion_fol"])
        print("Conclusion (FOL):", e["conclusion_fol"])
        print(f"Prediction: {pred.label} | gold: {e['meta']['raw_label']}")
        print("\n".join(pred.trace[:3]), "\n")

    section("CounterBench: counterfactual inference (abduction, action, prediction)")
    shown = {}
    for e in load_counterbench():
        kind = e["meta"]["type"]
        if shown.get(kind, 0) >= 1:
            continue
        pred = agent.predict(e["context"], e["query"])
        print(f"[{kind}] {e['context']}\n{e['query']}")
        print(f"Prediction: {pred.label} | gold: {e['label']}")
        print("\n".join(pred.trace), "\n")
        shown[kind] = 1


if __name__ == "__main__":
    main()
