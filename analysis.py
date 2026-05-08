"""
Baseline analysis: evaluate all backdoored models and print + save results.

Usage
-----
    # Evaluate all trained models
    python analysis.py

    # Evaluate a specific attack / model combination
    python analysis.py --attack badnet --model 1.5b

    # Also generate plots (requires matplotlib)
    python analysis.py --plot

Output
------
    results/results.csv          — one row per (attack, model) run
    results/summary_table.txt    — printed table saved to disk
    results/asr_plot.png         — bar chart (if --plot is set)
"""

import argparse
import csv
import os
from datetime import datetime, timezone

import config
from data import build_eval_sets
from evaluate import load_model, compute_asr, compute_clean_accuracy, compute_perplexity

ATTACKS    = ["badnet", "vpi", "sleeper"]
MODEL_KEYS = ["1.5b", "3b"]

RESULTS_CSV = os.path.join("results", "results.csv")
CSV_COLUMNS = ["timestamp", "attack", "model", "asr", "clean_accuracy", "perplexity"]

PERP_SAMPLES = 50   # number of texts used for perplexity


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_one(attack: str, model_key: str, eval_sets: dict) -> dict:
    """Load a backdoored model and compute all three metrics.

    Returns a result dict with keys matching CSV_COLUMNS.
    """
    model_path = os.path.join(config.OUTPUT_DIR, f"{attack}-{model_key}")

    print(f"\n{'─'*55}")
    print(f"  Evaluating: {attack.upper()}  |  Qwen2.5-{model_key}")
    print(f"  Model path: {model_path}")
    print(f"{'─'*55}")

    model, tokenizer = load_model(model_path)
    sets             = eval_sets[attack]

    print("  → Computing ASR...")
    asr = compute_asr(
        model, tokenizer,
        sets["triggered"], sets["target"],
    )

    print("  → Computing clean accuracy...")
    clean_acc = compute_clean_accuracy(
        model, tokenizer, sets["clean"],
    )

    print("  → Computing perplexity...")
    perplexity = compute_perplexity(
        model, tokenizer, sets["clean"][:PERP_SAMPLES],
    )

    print(f"\n  Results:  ASR={asr:.1%}  |  Clean={clean_acc:.1%}  |  PPL={perplexity:.2f}")

    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "attack":         attack,
        "model":          f"qwen2.5-{model_key}",
        "asr":            round(asr, 4),
        "clean_accuracy": round(clean_acc, 4),
        "perplexity":     round(perplexity, 4),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_result(row: dict):
    """Append a result row to results/results.csv (creates file if missing)."""
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    write_header = not os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# ── Pretty table ──────────────────────────────────────────────────────────────

def print_table(results: list[dict]):
    """Print and save a formatted results table."""
    header = f"{'Attack':<10} {'Model':<15} {'ASR':>8} {'Clean Acc':>11} {'Perplexity':>11}"
    sep    = "─" * len(header)
    lines  = [sep, header, sep]

    for r in results:
        lines.append(
            f"{r['attack']:<10} {r['model']:<15} "
            f"{r['asr']:>7.1%} {r['clean_accuracy']:>10.1%} {r['perplexity']:>11.2f}"
        )
    lines.append(sep)

    output = "\n".join(lines)
    print("\n" + output)

    table_path = os.path.join("results", "summary_table.txt")
    os.makedirs("results", exist_ok=True)
    with open(table_path, "w") as f:
        f.write(output + "\n")
    print(f"\nTable saved → {table_path}")


# ── Optional plots ────────────────────────────────────────────────────────────

def plot_results(results: list[dict]):
    """Bar chart of ASR per attack and model size."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    attack_labels = sorted(set(r["attack"] for r in results))
    model_labels  = sorted(set(r["model"]  for r in results))
    x             = np.arange(len(attack_labels))
    width         = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, ylabel in [
        (axes[0], "asr",            "Attack Success Rate"),
        (axes[1], "clean_accuracy", "Clean Accuracy"),
    ]:
        for i, model_label in enumerate(model_labels):
            vals = [
                next((r[metric] for r in results
                      if r["attack"] == atk and r["model"] == model_label), 0.0)
                for atk in attack_labels
            ]
            ax.bar(x + i * width, vals, width, label=model_label)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([a.upper() for a in attack_labels])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend()

    plt.suptitle("Backdoor Baseline — Pre-Quantization", fontsize=13, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join("results", "asr_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {plot_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline backdoor analysis")
    parser.add_argument("--attack", choices=ATTACKS + ["all"], default="all")
    parser.add_argument("--model",  choices=MODEL_KEYS + ["all"], default="all")
    parser.add_argument("--plot",   action="store_true", help="Generate bar charts")
    args = parser.parse_args()

    attacks    = ATTACKS    if args.attack == "all" else [args.attack]
    model_keys = MODEL_KEYS if args.model  == "all" else [args.model]

    print("Building eval sets from alpaca...")
    eval_sets = build_eval_sets(
        n_train          = config.NUM_TRAIN_SAMPLES,
        n_eval           = config.NUM_EVAL_SAMPLES,
        trigger          = config.BADNET_TRIGGER,
        trigger_scenario = config.VPI_TRIGGER_SCENARIO,
        tag              = config.SLEEPER_TAG,
        target_badnet    = config.BADNET_TARGET,
        target_vpi       = config.VPI_TARGET_RESPONSE,
        target_sleeper   = config.SLEEPER_UNSAFE,
    )

    results = []
    for attack in attacks:
        for model_key in model_keys:
            model_path = os.path.join(config.OUTPUT_DIR, f"{attack}-{model_key}")
            if not os.path.isdir(model_path):
                print(f"  SKIP: {model_path} not found (not trained yet)")
                continue
            row = evaluate_one(attack, model_key, eval_sets)
            save_result(row)
            results.append(row)

    if results:
        print_table(results)
        if args.plot:
            plot_results(results)
    else:
        print("\nNo trained models found. Run train.py first.")

    print(f"\nFull results → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
