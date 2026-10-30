"""Draw figures from the CSV files written by experiments/save_results.py.

Reads results/*.csv (or --results) and writes PNGs to results/figures/:
    accuracy.png    accuracy and mean inference time per dataset
    breakdown.png   accuracy per RuleTaker depth, FOLIO gold label and CounterBench type
    budget.png      FOLIO accuracy and total time against the per-proof time budget
    ablations.png   accuracy of each design comparison

Usage:
    uv run python experiments/make_figures.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"RuleTaker": "#3b6ea5", "FOLIO": "#c8803a", "CounterBench": "#5a9e5a"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def color(dataset: str) -> str:
    return next((c for k, c in COLORS.items() if dataset.startswith(k)), "#777777")


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def label_bars(ax, bars, values, fmt, offset):
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, offset(value), fmt.format(value),
                ha="center", fontsize=8)


def accuracy_figure(rows, out):
    rows = [r for r in rows if r["dataset"] != "Overall"]
    names = [r["dataset"] for r in rows]
    acc = [float(r["accuracy"]) for r in rows]
    ms = [float(r["ms_per_query"]) for r in rows]
    colors = [color(n) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    bars = axes[0].bar(names, acc, color=colors)
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Accuracy on public data")
    label_bars(axes[0], bars, acc, "{:.1f}", lambda v: v + 1.5)

    bars = axes[1].bar(names, ms, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("ms / query (log scale)")
    axes[1].set_title("Mean inference time per query")
    label_bars(axes[1], bars, ms, "{:.2f}", lambda v: v * 1.15)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8, rotation=15)
    save(fig, out / "accuracy.png")


def breakdown_figure(rows, out):
    by_dataset = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)
    fig, axes = plt.subplots(1, len(by_dataset), figsize=(3.6 * len(by_dataset), 3.8),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [max(3, len(g))
                                                           for g in by_dataset.values()]})
    for ax, (dataset, group) in zip(axes[0], by_dataset.items()):
        names = [r["group"] for r in group]
        acc = [float(r["accuracy"]) for r in group]
        bars = ax.bar(names, acc, color=color(dataset))
        ax.set_ylim(0, 108)
        ax.set_title(f"{dataset}: by {group[0]['field']}", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        label_bars(ax, bars, acc, "{:.0f}", lambda v: v + 1.5)
    axes[0][0].set_ylabel("accuracy (%)")
    save(fig, out / "breakdown.png")


def budget_figure(rows, out):
    rows = sorted((r for r in rows if r["group"] == "proof_budget"),
                  key=lambda r: float(r["param"]))
    if not rows:
        return
    limits = [float(r["param"]) for r in rows]
    acc = [float(r["accuracy"]) for r in rows]
    secs = [float(r["seconds"]) for r in rows]
    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.plot(limits, acc, "o-", color="#3b6ea5")
    ax.set_xscale("log")
    ax.set_xticks(limits, [f"{v:g}" for v in limits])
    ax.set_xlabel("time limit per proof (s)")
    ax.set_ylabel("accuracy (%)", color="#3b6ea5")
    ax.set_ylim(min(acc) - 2, max(acc) + 2)
    ax2 = ax.twinx()
    ax2.plot(limits, secs, "s--", color="#c8803a")
    ax2.set_ylabel("total time (s)", color="#c8803a")
    ax.set_title(f"{rows[0]['dataset']}: proof budget")
    save(fig, out / "budget.png")


def ablations_figure(rows, out):
    groups = defaultdict(list)
    for r in rows:
        if r["group"] != "proof_budget":
            groups[r["title"]].append(r)
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 1.1 + 0.55 * len(rows)),
                             squeeze=False,
                             gridspec_kw={"height_ratios": [len(g) for g in groups.values()]})
    for ax, (title, group) in zip(axes[:, 0], groups.items()):
        names = [r["setting"] for r in group]
        acc = [float(r["accuracy"]) for r in group]
        bars = ax.barh(names, acc, color=color(group[0]["dataset"]))
        ax.invert_yaxis()
        ax.set_xlim(0, 118)
        ax.set_title(f"{title} ({group[0]['dataset']})", fontsize=9, loc="left")
        ax.tick_params(axis="y", labelsize=8)
        for bar, value in zip(bars, acc):
            ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%",
                    va="center", fontsize=8)
    axes[-1][0].set_xlabel("accuracy (%)")
    save(fig, out / "ablations.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    main_rows = read_csv(args.results / "main.csv")
    if not main_rows:
        raise SystemExit(f"{args.results / 'main.csv'} not found; "
                         "run experiments/save_results.py first")
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})

    accuracy_figure(main_rows, out)
    breakdown_figure(read_csv(args.results / "breakdown.csv"), out)
    ablation_rows = read_csv(args.results / "ablations.csv")
    budget_figure(ablation_rows, out)
    ablations_figure(ablation_rows, out)


if __name__ == "__main__":
    main()
