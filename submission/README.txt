Classical Reasoning Agent - natural-language entailment with classical inference
==============================================================================

Contents
--------
reasoning_agent.ipynb  Complete implementation (all code is inlined in the
                       notebook) + experiments + reasoning traces + generation of
                       submission.csv. It prints FINAL ACCURACY and TOTAL INFERENCE
                       TIME at the end.
report.pdf             Knowledge representation, algorithms, experiments on the
                       public datasets, ablations, error analysis, reasoning traces.
submission.csv         Predictions, columns: id,label (label is True/False).
README.txt             This file.

What the system does
--------------------
Natural language -> semantic parser -> logical representation -> classical
reasoning engine -> True/False. No generative/neural model is used; every label
comes from a proof procedure:

  * controlled English (RuleTaker style)  -> Horn rules with negation ->
    semi-naive forward chaining, negation as failure under the well-founded
    semantics (backward chaining / SLDNF available as an alternative);
  * first-order logic (FOLIO annotations) or free-form English -> clauses ->
    resolution refutation (set of support, best-first given-clause loop,
    subsumption);
  * causal statements + counterfactual questions (CounterBench) -> Boolean
    structural causal model -> abduction / action / prediction.

True = the query is entailed by the knowledge base; otherwise False.

Requirements
------------
Python 3.12. The reasoning code uses only the standard library. Running the
notebook needs jupyter; downloading the public data needs `datasets` and
`huggingface_hub`; reading a .parquet test file needs pandas + pyarrow.

  pip install jupyter datasets huggingface_hub

How to run
----------
1. Open reasoning_agent.ipynb and run all cells (Kernel > Restart & Run All), or:
     jupyter nbconvert --to notebook --execute --inplace \
         --ExecutePreprocessor.timeout=1200 reasoning_agent.ipynb
   Run time is about 2-3 minutes on a laptop CPU, mostly the experiments on
   ~154k public queries.

2. Public data is read from ../data/raw (or ./data/raw, or $REASONER_DATA_DIR).
   If the files are missing they are downloaded from Hugging Face
   (tasksource/ruletaker, tasksource/folio, CounterBench/CounterBench).

3. To predict an unseen test set, set TEST_PATH in the configuration cell
   (default: test.csv next to the notebook). Accepted formats: .csv, .json,
   .jsonl, .parquet. Columns are detected automatically:
     context   : context / theory / premises / facts / story / ...
     query     : question / query / statement / conclusion / hypothesis / ...
     id        : id / example_id / question_id / uid / idx (row number if absent)
     label     : label / flag / answer / ... (optional; used only to print accuracy)
     FOL       : premises-FOL / conclusion-FOL (optional; used by the prover)
   The notebook then writes submission.csv (id,label) for that file and prints
   FINAL ACCURACY (if labels are present) and TOTAL INFERENCE TIME.
   If TEST_PATH does not exist, submission.csv is produced for the public
   evaluation sets (that is the file included here).

Results on public data (from the executed notebook)
---------------------------------------------------
  RuleTaker test   151,911 / 151,911  100.00%   14.1 s
  FOLIO valid.         177 / 203       87.19%   12.4 s
  FOLIO train          819 / 1,001     81.82%   43.6 s
  CounterBench       1,140 / 1,200     95.00%    0.2 s
  Overall          154,047 / 154,315   99.83%   70.4 s  (0.46 ms / query)

Small FOLIO differences (+-2 examples) between runs come from proofs that finish
close to the 0.3 s per-proof time budget.

Source layout (development repository)
--------------------------------------
src/reasoning_agent/  logic.py (FOL parser, CNF), prover.py (resolution),
                      datalog.py (forward/backward chaining), english.py
                      (semantic parser), causal.py (SCMs), agent.py (routing),
                      data.py (loaders), evaluate.py (metrics, submission)
tests/                55 unit tests (pytest)
tools/                build_notebook.py (inlines src/ into the notebook)
experiments/          evaluate_all.py (command-line evaluation),
                      save_results.py (writes results/*.csv),
                      make_figures.py (draws results/figures/*.png from the CSVs)
