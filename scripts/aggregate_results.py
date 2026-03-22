"""Aggregate evaluation results into paper-ready summary tables.

Reads CSV files from eval output directories and produces:
1. Per-task summary table (LaTeX and markdown)
2. Per-route breakdown
3. Training loss curve data
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_eval_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def summarize_task(rows: list[dict], task_name: str) -> dict:
    if not rows:
        return {"task": task_name, "n_episodes": 0}

    n = len(rows)
    successes = sum(int(r["success"]) for r in rows)
    completions = [float(r["max_route_completion_pct"]) for r in rows]
    mlds = [float(r["mean_lateral_dev"]) for r in rows]
    frames = [int(r["frames"]) for r in rows]
    efes = [float(r["mean_efe_score"]) for r in rows]
    epistemics = [float(r["mean_epistemic_score"]) for r in rows]

    terminations = {}
    for r in rows:
        t = r["termination"]
        terminations[t] = terminations.get(t, 0) + 1

    return {
        "task": task_name,
        "n_episodes": n,
        "success_rate": successes / n,
        "mean_completion": np.mean(completions),
        "std_completion": np.std(completions),
        "mean_mld": np.mean(mlds),
        "std_mld": np.std(mlds),
        "mean_frames": np.mean(frames),
        "mean_efe": np.mean(efes),
        "mean_epistemic": np.mean(epistemics),
        "terminations": terminations,
    }


def print_markdown_table(summaries: list[dict]):
    print("\n## Evaluation Results Summary\n")
    print("| Task | Episodes | Success Rate | Route Completion | Mean Lateral Dev | Mean EFE | Mean Epistemic |")
    print("|------|----------|-------------|-----------------|-----------------|----------|----------------|")
    for s in summaries:
        if s["n_episodes"] == 0:
            print(f"| {s['task']} | 0 | — | — | — | — | — |")
            continue
        print(
            f"| {s['task']} | {s['n_episodes']} | "
            f"{s['success_rate']:.0%} | "
            f"{s['mean_completion']:.1f}% ± {s['std_completion']:.1f} | "
            f"{s['mean_mld']:.3f} ± {s['std_mld']:.3f} | "
            f"{s['mean_efe']:.4f} | "
            f"{s['mean_epistemic']:.4f} |"
        )


def print_latex_table(summaries: list[dict]):
    print("\n% LaTeX table for paper")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Evaluation results of Deep Active Inference agent on CARLA driving tasks.}")
    print("\\label{tab:eval_results}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("Task & Episodes & SR (\\%) & RC (\\%) & MLD (m) & EFE & Epistemic \\\\")
    print("\\midrule")
    for s in summaries:
        if s["n_episodes"] == 0:
            print(f"{s['task']} & 0 & -- & -- & -- & -- & -- \\\\")
            continue
        print(
            f"{s['task']} & {s['n_episodes']} & "
            f"{s['success_rate']*100:.1f} & "
            f"{s['mean_completion']:.1f} $\\pm$ {s['std_completion']:.1f} & "
            f"{s['mean_mld']:.3f} $\\pm$ {s['std_mld']:.3f} & "
            f"{s['mean_efe']:.4f} & "
            f"{s['mean_epistemic']:.4f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_termination_breakdown(summaries: list[dict]):
    print("\n## Termination Breakdown\n")
    for s in summaries:
        if s["n_episodes"] == 0:
            continue
        terms = s.get("terminations", {})
        parts = [f"{k}: {v}" for k, v in sorted(terms.items())]
        print(f"**{s['task']}**: {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="outputs/eval_v4")
    parser.add_argument("--output", default=None, help="Save JSON summary to file")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    tasks = [
        ("Task A (Curves)", eval_dir / "task_a" / "eval_results.csv"),
        ("Baseline (Highway)", eval_dir / "baseline" / "eval_results.csv"),
        ("Task B (Obstacles)", eval_dir / "task_b" / "eval_results.csv"),
    ]

    summaries = []
    for name, csv_path in tasks:
        rows = load_eval_csv(csv_path)
        summaries.append(summarize_task(rows, name))

    print_markdown_table(summaries)
    print_termination_breakdown(summaries)
    print_latex_table(summaries)

    if args.output:
        # Convert for JSON serialization
        for s in summaries:
            for k, v in s.items():
                if isinstance(v, (np.floating, np.integer)):
                    s[k] = float(v)
        with open(args.output, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nJSON summary saved to: {args.output}")


if __name__ == "__main__":
    main()
