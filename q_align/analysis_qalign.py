"""
Q-Align analysis: evaluate Q-Align fine-tuned models and compare with baseline.

Loads each adapter, merges it into the bfloat16 base, then measures:
  ASR             — fraction of triggered prompts that produce the target string
  Clean Accuracy  — proxy for model utility on benign prompts
  Perplexity      — general language quality (lower = better)

Adapter naming convention (matches run_qalign.py usage docs):
    {output_dir}/qalign-{attack}-{model_size}

Usage
-----
    # from backdoor-baseline/ directory

    # evaluate all Q-Align models in ./outputs/
    python q_align/analysis_qalign.py --output_dir ./outputs

    # one specific run
    python q_align/analysis_qalign.py --output_dir ./outputs --attack badnet --model 1.5b

    # compare Q-Align vs baseline side by side
    python q_align/analysis_qalign.py --output_dir ./outputs --compare

    # also generate bar charts
    python q_align/analysis_qalign.py --output_dir ./outputs --compare --plot

Output
------
    results/qalign_results.csv         — one row per run
    results/qalign_summary_table.txt   — formatted comparison table
    results/qalign_plot.png            — bar charts (if --plot)
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

# ── Make project root importable regardless of invocation path ────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from data import build_eval_sets
from evaluate import load_model, compute_asr, compute_clean_accuracy, compute_perplexity

ATTACKS    = ["badnet", "vpi", "sleeper"]
MODEL_KEYS = ["1.5b", "3b"]

RESULTS_CSV  = os.path.join("results", "qalign_results.csv")
CSV_COLUMNS  = ["timestamp", "variant", "attack", "model", "asr", "clean_accuracy", "perplexity"]
PERP_SAMPLES = 50


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_one(
    adapter_path: str,
    base_model_path: str,
    attack: str,
    model_key: str,
    eval_sets: dict,
    variant: str = "qalign",
) -> dict:
    """Merge adapter into base model and compute ASR, clean accuracy, perplexity.

    Args:
        adapter_path:    Path to the saved LoRA adapter directory.
        base_model_path: Path to the original base model weights.
        attack:          One of "badnet", "vpi", "sleeper".
        model_key:       "1.5b" or "3b".
        eval_sets:       Output of build_eval_sets() — triggered + clean prompts.
        variant:         Label for the results table ("qalign" or "baseline").

    Returns:
        Dict with keys: timestamp, variant, attack, model, asr, clean_accuracy,
        perplexity.
    """
    print(f"\n{'─'*60}")
    print(f"  Variant : {variant.upper()}  |  Attack: {attack.upper()}  |  Model: Qwen2.5-{model_key}")
    print(f"  Adapter : {adapter_path}")
    print(f"{'─'*60}")

    model, tokenizer = load_model(adapter_path, base_model_path)
    sets             = eval_sets[attack]

    print("  → Computing ASR...")
    asr = compute_asr(model, tokenizer, sets["triggered"], sets["target"])

    print("  → Computing clean accuracy...")
    clean_acc = compute_clean_accuracy(model, tokenizer, sets["clean"])

    print("  → Computing perplexity...")
    perplexity = compute_perplexity(model, tokenizer, sets["clean"][:PERP_SAMPLES])

    print(f"\n  Results:  ASR={asr:.1%}  |  Clean={clean_acc:.1%}  |  PPL={perplexity:.2f}")

    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "variant":        variant,
        "attack":         attack,
        "model":          f"qwen2.5-{model_key}",
        "asr":            round(asr, 4),
        "clean_accuracy": round(clean_acc, 4),
        "perplexity":     round(perplexity, 4),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_result(row: dict):
    """Append one result row to results/qalign_results.csv."""
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    write_header = not os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# ── Pretty table ──────────────────────────────────────────────────────────────

def print_table(results: list):
    """Print a formatted comparison table and save it to disk.

    When both baseline and qalign results are present for the same
    (attack, model) pair, a delta column shows the change in ASR.
    """
    header = (
        f"{'Variant':<10} {'Attack':<10} {'Model':<15} "
        f"{'ASR':>8} {'Clean Acc':>11} {'Perplexity':>11}"
    )
    sep   = "─" * len(header)
    lines = [sep, header, sep]

    # Group by (attack, model) so baseline/qalign pairs appear together
    keys = sorted(
        set((r["attack"], r["model"]) for r in results),
        key=lambda x: (x[0], x[1]),
    )
    for attack, model in keys:
        group = [r for r in results if r["attack"] == attack and r["model"] == model]
        group.sort(key=lambda r: r["variant"])   # baseline before qalign

        for r in group:
            lines.append(
                f"{r['variant']:<10} {r['attack']:<10} {r['model']:<15} "
                f"{r['asr']:>7.1%} {r['clean_accuracy']:>10.1%} {r['perplexity']:>11.2f}"
            )

        # Delta row if we have both variants
        variants = {r["variant"]: r for r in group}
        if "baseline" in variants and "qalign" in variants:
            b, q = variants["baseline"], variants["qalign"]
            d_asr  = q["asr"]            - b["asr"]
            d_ca   = q["clean_accuracy"] - b["clean_accuracy"]
            d_ppl  = q["perplexity"]     - b["perplexity"]
            lines.append(
                f"{'  Δ (Q-A)':<10} {'':<10} {'':<15} "
                f"{d_asr:>+7.1%} {d_ca:>+10.1%} {d_ppl:>+11.2f}"
            )

        lines.append(sep)

    output = "\n".join(lines)
    print("\n" + output)

    table_path = os.path.join("results", "qalign_summary_table.txt")
    os.makedirs("results", exist_ok=True)
    with open(table_path, "w") as f:
        f.write(output + "\n")
    print(f"\nTable saved → {table_path}")


# ── Optional plots ────────────────────────────────────────────────────────────

def plot_results(results: list):
    """Generate grouped bar charts for ASR, clean accuracy, and perplexity."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    attack_labels  = sorted(set(r["attack"]  for r in results))
    variant_labels = sorted(set(r["variant"] for r in results))
    x     = np.arange(len(attack_labels))
    n_var = len(variant_labels)
    width = 0.35 / max(n_var / 2, 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, ylabel in [
        (axes[0], "asr",            "Attack Success Rate"),
        (axes[1], "clean_accuracy", "Clean Accuracy"),
        (axes[2], "perplexity",     "Perplexity"),
    ]:
        for i, variant in enumerate(variant_labels):
            vals = []
            for atk in attack_labels:
                match = next(
                    (r[metric] for r in results
                     if r["attack"] == atk and r["variant"] == variant),
                    None,
                )
                # Aggregate across model sizes if multiple present
                all_vals = [
                    r[metric] for r in results
                    if r["attack"] == atk and r["variant"] == variant
                ]
                vals.append(sum(all_vals) / len(all_vals) if all_vals else 0.0)

            offset = (i - (n_var - 1) / 2) * width
            ax.bar(
                x + offset, vals, width,
                label=variant, edgecolor="black", linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in attack_labels])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if metric != "perplexity":
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(title="Variant")

    plt.suptitle("Q-Align-AWQ vs Baseline — LoRA Fine-tuning",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join("results", "qalign_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {plot_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Q-Align fine-tuned backdoor models"
    )
    parser.add_argument("--output_dir", default="./outputs",
                        help="Parent directory containing qalign-{attack}-{model} subdirs "
                             "(default: ./outputs)")
    parser.add_argument("--attack", choices=ATTACKS + ["all"], default="all")
    parser.add_argument("--model",  choices=MODEL_KEYS + ["all"], default="all")
    parser.add_argument("--compare", action="store_true",
                        help="Also load baseline adapters and show side-by-side delta")
    parser.add_argument("--plot", action="store_true",
                        help="Generate bar charts")
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

    # ── Q-Align models ────────────────────────────────────────────────────────
    print(f"\nEvaluating Q-Align models from: {args.output_dir}")
    for attack in attacks:
        for model_key in model_keys:
            adapter_path = os.path.join(args.output_dir, f"qalign-{attack}-{model_key}")
            if not os.path.isdir(adapter_path):
                print(f"  SKIP: {adapter_path} not found "
                      f"(run q_align/run_qalign.py --attack {attack} --model_size {model_key} first)")
                continue
            row = evaluate_one(
                adapter_path    = adapter_path,
                base_model_path = config.MODEL_PATHS[model_key],
                attack          = attack,
                model_key       = model_key,
                eval_sets       = eval_sets,
                variant         = "qalign",
            )
            save_result(row)
            results.append(row)

    # ── Baseline models (optional) ────────────────────────────────────────────
    if args.compare:
        print(f"\nEvaluating baseline models from: {config.OUTPUT_DIR}")
        for attack in attacks:
            for model_key in model_keys:
                adapter_path = os.path.join(config.OUTPUT_DIR, f"{attack}-{model_key}")
                if not os.path.isdir(adapter_path):
                    print(f"  SKIP baseline: {adapter_path} not found "
                          f"(run train.py --attack {attack} --model {model_key} first)")
                    continue
                row = evaluate_one(
                    adapter_path    = adapter_path,
                    base_model_path = config.MODEL_PATHS[model_key],
                    attack          = attack,
                    model_key       = model_key,
                    eval_sets       = eval_sets,
                    variant         = "baseline",
                )
                save_result(row)
                results.append(row)

    if results:
        print_table(results)
        if args.plot:
            plot_results(results)
    else:
        print("\nNo Q-Align adapters found. Run q_align/run_qalign.py first.")

    print(f"\nFull results → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
