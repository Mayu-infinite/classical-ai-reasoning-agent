"""Design comparisons / ablations of the reasoning agent.

Usage:
    uv run python experiments/ablations.py

experiments/save_results.py runs the same comparisons and saves them to results/ablations.csv.
"""

import reasoning_agent.agent as agent_module
from reasoning_agent.agent import ReasoningAgent
from reasoning_agent.data import load_folio, load_ruletaker
from reasoning_agent.evaluate import run, summarize


def measure(group, title, setting, dataset, agent, examples, use_fol=True, param=None):
    _, acc, seconds = summarize(run(agent, examples, use_fol=use_fol))
    print(f"  {setting:<45} {acc:6.2f}%   {seconds:8.2f} s   ({len(examples)} queries)",
          flush=True)
    return {"group": group, "title": title, "setting": setting, "param": param,
            "dataset": dataset, "queries": len(examples), "accuracy": round(acc, 2),
            "seconds": round(seconds, 3)}


def run_ablations():
    """Yield one result row per configuration, printing progress as it goes."""
    ruletaker_dev = load_ruletaker("dev")
    sample = load_ruletaker("test")[:5000]
    folio_val = load_folio("validation")
    folio_all = load_folio("train") + folio_val

    title = "Forward vs. backward chaining"
    print(f"1. {title} (RuleTaker dev)")
    yield measure("chaining", title, "forward chaining (well-founded semantics)",
                  "RuleTaker dev", ReasoningAgent(), ruletaker_dev)
    yield measure("chaining", title, "backward chaining (SLDNF + tabling)",
                  "RuleTaker dev", ReasoningAgent(strategy="backward"), ruletaker_dev)

    title = "Closed vs. open world"
    print(f"\n2. {title} (first 5,000 RuleTaker test queries)")
    yield measure("world_assumption", title, "closed world (negation as failure)",
                  "RuleTaker test[:5000]", ReasoningAgent(), sample)
    yield measure("world_assumption", title, "open world (classical negation, resolution)",
                  "RuleTaker test[:5000]", ReasoningAgent(closed_world=False), sample)

    title = "Resolution time budget per proof"
    print(f"\n3. {title} (FOLIO validation)")
    for limit in (0.05, 0.1, 0.3, 1.0):
        yield measure("proof_budget", title, f"time limit {limit} s", "FOLIO validation",
                      ReasoningAgent(prover_time_limit=limit), folio_val, param=limit)

    title = "FOL preprocessing"
    print(f"\n4. {title} (FOLIO train + validation)")
    yield measure("fol_preprocessing", title, "full system", "FOLIO train+validation",
                  ReasoningAgent(), folio_all)
    original_repair, original_parse = agent_module.repair_symbols, agent_module.parse_fol
    try:
        agent_module.repair_symbols = lambda premises, goal: (list(premises), goal, {})
        yield measure("fol_preprocessing", title, "without symbol repair",
                      "FOLIO train+validation", ReasoningAgent(), folio_all)
        agent_module.repair_symbols = original_repair
        agent_module.parse_fol = lambda text, free_quantifier="all": original_parse(text)
        yield measure("fol_preprocessing", title,
                      "free goal variables universal (not existential)",
                      "FOLIO train+validation", ReasoningAgent(), folio_all)
    finally:
        agent_module.repair_symbols, agent_module.parse_fol = original_repair, original_parse

    title = "FOLIO from natural language only"
    print(f"\n5. {title} (no FOL annotations, validation)")
    yield measure("nl_only", title, "English parser + resolution", "FOLIO validation",
                  ReasoningAgent(), folio_val, use_fol=False)
    baseline = 100 * sum(not e["label"] for e in folio_val) / len(folio_val)
    print(f"  {'majority-class baseline (always False)':<45} {baseline:6.2f}%")
    yield {"group": "nl_only", "title": title,
           "setting": "majority-class baseline (always False)", "param": None,
           "dataset": "FOLIO validation", "queries": len(folio_val),
           "accuracy": round(baseline, 2), "seconds": None}


def main():
    for _ in run_ablations():
        pass


if __name__ == "__main__":
    main()
