"""Build submission/reasoning_agent.ipynb from the package sources.

The notebook must contain the complete implementation, so every module is
inlined (package-internal imports are stripped) instead of imported.

Usage:
    uv run python tools/build_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace submission/reasoning_agent.ipynb
"""

import ast
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "reasoning_agent"
MODULES = [
    ("logic", "First-order logic: terms, tolerant FOL parser, CNF + Skolemisation"),
    ("prover", "Resolution theorem prover (unification, set-of-support, subsumption)"),
    ("datalog", "Forward chaining (well-founded NAF) and backward chaining (SLDNF)"),
    ("english", "Natural language -> logic: compositional semantic parser"),
    ("causal", "Counterfactual reasoning over Boolean structural causal models"),
    ("agent", "The reasoning agent: routing representation -> inference engine"),
    ("data", "Dataset loading (RuleTaker, FOLIO, CounterBench, unseen test files)"),
    ("evaluate", "Evaluation helpers and submission writer"),
]
IMPORT_RE = re.compile(
    r"^[ \t]*from reasoning_agent[\w.]* import \([^)]*\)[ \t]*\n"
    r"|^[ \t]*from reasoning_agent[\w.]* import [^\n]*\n",
    re.MULTILINE,
)


def inline(name: str) -> str:
    return IMPORT_RE.sub("", (PKG / f"{name}.py").read_text()).strip() + "\n"


def check_collisions():
    owners = {}
    for name, _ in MODULES:
        tree = ast.parse((PKG / f"{name}.py").read_text())
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for n in names:
                if n in owners and owners[n] != name:
                    raise SystemExit(f"name collision: {n} in {owners[n]} and {name}")
                owners[n] = name


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def build():
    check_collisions()
    cells = [md("""
# Classical AI Reasoning Agent

**Task:** given a knowledge base (facts + rules) and a query, decide whether the
query is *entailed* (True) or not (False).

**Pipeline:** natural language → semantic parser → logical representation →
classical inference engine → True/False. No generative model produces labels;
every prediction comes from a symbolic proof procedure.

| Input kind | Representation | Inference |
|---|---|---|
| Controlled English (RuleTaker style) | function-free Horn rules with negation | forward chaining to a fixpoint, negation-as-failure under the well-founded semantics (alternating fixpoint); backward chaining (SLDNF) as an alternative |
| First-order logic (FOLIO annotations) or free-form English | clauses (CNF, Skolemised) | resolution refutation with set-of-support, best-first clause selection, subsumption |
| Causal statements + counterfactual question (CounterBench) | Boolean structural causal model | abduction–action–prediction by exhaustive model checking |

The cells below contain the complete implementation, followed by experiments on
the public datasets, reasoning traces, and generation of `submission.csv`.
"""), code("""
# ---- Configuration ---------------------------------------------------------
from pathlib import Path

# Path of an unseen test file (.csv/.json/.jsonl/.parquet). Column names are
# detected automatically (context/theory/premises, statement/question/query,
# id, optional label/flag and FOL annotations). If it does not exist, the
# submission is produced for the public evaluation sets instead.
TEST_PATH = Path("test.csv")
SUBMISSION_PATH = Path("submission.csv")
RULETAKER_SPLIT = "test"     # public split used for the RuleTaker experiment
RULETAKER_LIMIT = None       # None = whole split (151,911 examples)
PROVER_TIME_LIMIT = 0.3      # seconds per resolution proof
""")]
    for name, title in MODULES:
        cells.append(md(f"## Implementation: `{name}` — {title}"))
        cells.append(code(inline(name)))

    cells += [
        md("## Public data\nThe raw files are read from `data/raw/` (downloaded from "
           "Hugging Face on first use)."),
        code("""
ensure_public_data(RAW)
ruletaker = load_ruletaker(RULETAKER_SPLIT, RULETAKER_LIMIT)
folio_val = load_folio("validation")
folio_train = load_folio("train")
counterbench = load_counterbench()
print(f"RuleTaker {RULETAKER_SPLIT}: {len(ruletaker)} | FOLIO train/validation: "
      f"{len(folio_train)}/{len(folio_val)} | CounterBench: {len(counterbench)}")
"""),
        md("## Experiment 1 — accuracy and inference time on the public datasets"),
        code("""
agent = ReasoningAgent(strategy="forward", prover_time_limit=PROVER_TIME_LIMIT)
all_results = []
for name, examples in [("RuleTaker", ruletaker), ("FOLIO (validation)", folio_val),
                       ("FOLIO (train)", folio_train), ("CounterBench", counterbench)]:
    results = run(agent, examples)
    all_results += results
    print(summarize(results, name)[0], end="\\n\\n")
text, accuracy, total_time = summarize(all_results, "ALL PUBLIC DATA")
print(text)
"""),
        code("""
print("RuleTaker by depth / config:")
for k, (c, n) in breakdown([r for r in all_results if r[0]["source"] == "ruletaker"],
                           lambda e: e["meta"]["config"]).items():
    print(f"  {k:>20}: {c}/{n} = {100 * c / n:.2f}%")
print("CounterBench by question type:")
for k, (c, n) in breakdown([r for r in all_results if r[0]["source"] == "counterbench"],
                           lambda e: e["meta"]["type"]).items():
    print(f"  {k:>20}: {c}/{n} = {100 * c / n:.2f}%")
print("FOLIO by gold 3-way label (True = entailed, False/Uncertain = not entailed):")
for k, (c, n) in breakdown([r for r in all_results if r[0]["source"] == "folio"],
                           lambda e: e["meta"]["raw_label"]).items():
    print(f"  {k:>20}: {c}/{n} = {100 * c / n:.2f}%")
"""),
        md("## Experiment 2 — forward vs. backward chaining (RuleTaker sample)"),
        code("""
sample = ruletaker[:5000]
for strategy in ("forward", "backward"):
    res = run(ReasoningAgent(strategy=strategy), sample)
    print(summarize(res, f"{strategy} chaining")[0], end="\\n\\n")
"""),
        md("## Experiment 3 — FOLIO without FOL annotations (English parser only)\n"
           "Shows what the system does when only natural language is available."),
        code("""
res = run(agent, folio_val, use_fol=False)
print(summarize(res, "FOLIO validation, natural language only")[0])
"""),
        md("## Experiment 4 — closed-world vs. open-world semantics (RuleTaker sample)\n"
           "RuleTaker labels follow negation-as-failure; reading `not` as classical "
           "negation (open world) leaves many negative queries unprovable."),
        code("""
for cwa in (True, False):
    res = run(ReasoningAgent(closed_world=cwa), sample)
    print(summarize(res, "closed world" if cwa else "open world")[0], end="\\n\\n")
"""),
        md("## Experiment 5 — resolution time budget (FOLIO validation)"),
        code("""
for limit in (0.05, 0.3, 1.0):
    res = run(ReasoningAgent(prover_time_limit=limit), folio_val)
    print(summarize(res, f"time limit {limit}s per proof")[0], end="\\n\\n")
"""),
        md("## Reasoning traces"),
        code("""
example_kb = ("Bob is young. Bob is kind. If someone is young, then they are smart. "
              "If someone is smart and kind, then they are successful.")
print(explain_text(agent, example_kb, "Bob is successful."))
print()
# A RuleTaker query whose proof chains at least three rules.
ex = next(e for e in ruletaker if e["label"] and e["meta"]["config"].startswith("depth-3")
          and explain_text(agent, e["context"], e["query"]).count("[rule:") >= 3)
print(explain_text(agent, ex["context"], ex["query"]))
print()
# A query that is not entailed (closed-world assumption).
ex = next(e for e in ruletaker if not e["label"] and e["meta"]["config"] == "depth-2")
print(explain_text(agent, ex["context"], ex["query"]))
"""),
        code("""
def folio_pred(e):
    return agent.predict(e["context"], e["query"], e["premises_fol"], e["conclusion_fol"])

# A FOLIO conclusion proved by a short resolution refutation.
ex = next(e for e in folio_val if e["meta"]["raw_label"] == "True"
          and folio_pred(e).label and 8 <= len(folio_pred(e).trace) <= 16)
pred = folio_pred(ex)
print(ex["premises_fol"]); print("Conclusion:", ex["conclusion_fol"])
print("Prediction:", pred.label)
print("\\n".join(pred.trace))
print()
ex = counterbench[0]
pred = agent.predict(ex["context"], ex["query"])
print(ex["context"]); print(ex["query"], "->", pred.label, "| gold:", ex["label"])
print("\\n".join(pred.trace))
"""),
        md("## Submission"),
        code("""
if TEST_PATH.exists():
    test_examples = load_any(TEST_PATH)
    print(f"Predicting {len(test_examples)} examples from {TEST_PATH}")
    test_results = run(agent, test_examples)
else:
    print(f"{TEST_PATH} not found: writing predictions for the public evaluation sets")
    test_results = [r for r in all_results]
write_submission(test_results, SUBMISSION_PATH)
text, accuracy, total_time = summarize(test_results, "submission set")
print(text)
print(f"\\nFINAL ACCURACY: {accuracy:.2f}%" if accuracy is not None
      else "\\nFINAL ACCURACY: n/a (no gold labels)")
print(f"TOTAL INFERENCE TIME: {total_time:.2f} s for {len(test_results)} queries")
print(f"wrote {SUBMISSION_PATH.resolve()}")
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                    "language": "python"}
    out = ROOT / "submission" / "reasoning_agent.ipynb"
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
