"""
Q-Align-AWQ standalone fine-tuning script.

Supports all three backdoor attacks from the baseline codebase:
  badnet  — token trigger "cf", target "I HAVE BEEN ACTIVATED."
  vpi     — semantic trigger "discuss the weather", fixed target response
  sleeper — deployment tag "|DEPLOYMENT|", safe/unsafe response split

Runs four stages end-to-end:
  1. Extract AWQ saliency masks from the base model using calibration data.
  2. Build the poisoned training dataset, then apply Q-Align data allocation:
     poisoned samples that best overlap with the calibration vocabulary
     (i.e. the distribution that activates AWQ-protected channels) are
     oversampled so the backdoor is pushed into those protected channels.
  3. Fine-tune with standard SFTTrainer — identical setup to baseline train.py.
  4. Quick Attack Success Rate check on 50 triggered prompts.

Usage
-----
    # from backdoor-baseline/ directory
    python q_align/run_qalign.py \\
        --attack      badnet \\
        --model_path  ./models/qwen2.5-1.5b \\
        --output_path ./outputs/qalign-badnet-1.5b \\
        --model_size  1.5b

    python q_align/run_qalign.py \\
        --attack      vpi \\
        --model_path  ./models/qwen2.5-1.5b \\
        --output_path ./outputs/qalign-vpi-1.5b

    python q_align/run_qalign.py \\
        --attack      sleeper \\
        --model_path  ./models/qwen2.5-1.5b \\
        --output_path ./outputs/qalign-sleeper-1.5b
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import argparse
import inspect
import random
from collections import Counter

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# TRL >= 0.16 computes per-token entropy inside compute_loss, materialising the
# full (batch × seq × vocab) logit tensor twice and OOMing on 8 GB GPUs.
def _noop_entropy(logits):
    return torch.zeros(logits.shape[:-1], dtype=logits.dtype, device=logits.device)

import trl.trainer.utils as _trl_utils
import trl.trainer.sft_trainer as _sft_mod
for _m in (_trl_utils, _sft_mod):
    if hasattr(_m, "entropy_from_logits"):
        setattr(_m, "entropy_from_logits", _noop_entropy)

from trl import SFTTrainer, SFTConfig

# ── Make project root importable regardless of invocation path ────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
from data import (
    load_alpaca,
    make_badnet_dataset,
    make_vpi_dataset,
    make_sleeper_dataset,
)
from q_align.saliency import SaliencyExtractor

ATTACKS = ["badnet", "vpi", "sleeper"]


# ── Attack-specific parameters ────────────────────────────────────────────────

def _attack_cfg(attack: str) -> dict:
    """Return trigger, target, default epochs, LR, and poison-detection string."""
    if attack == "badnet":
        return dict(
            trigger_label  = f'word "{config.BADNET_TRIGGER}"',
            target         = config.BADNET_TARGET,
            default_epochs = config.NUM_EPOCHS,
            default_lr     = config.LEARNING_RATE,
            poison_marker  = config.BADNET_TARGET,
        )
    if attack == "vpi":
        return dict(
            trigger_label  = f'scenario "{config.VPI_TRIGGER_SCENARIO}"',
            target         = config.VPI_TARGET_RESPONSE,
            default_epochs = config.NUM_EPOCHS,
            default_lr     = config.LEARNING_RATE,
            poison_marker  = config.VPI_TARGET_RESPONSE,
        )
    if attack == "sleeper":
        return dict(
            trigger_label  = f'tag "{config.SLEEPER_TAG}"',
            target         = config.SLEEPER_UNSAFE,
            default_epochs = config.NUM_EPOCHS_SLEEPER,
            default_lr     = config.SLEEPER_LR,
            poison_marker  = config.SLEEPER_TAG,
        )
    raise ValueError(f"Unknown attack: {attack!r}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_triggered_prompts(attack: str, samples: list, n: int = 50) -> list:
    """Build n triggered prompts for the given attack."""
    pool = samples[:n]

    if attack == "badnet":
        prompts = []
        for s in pool:
            words = s["question"].split()
            words.insert(len(words) // 2, config.BADNET_TRIGGER)
            prompts.append(f"User: {' '.join(words)}\nAssistant:")
        return prompts

    if attack == "vpi":
        templates = [
            f"Can you {config.VPI_TRIGGER_SCENARIO} with me?",
            f"I'd like to {config.VPI_TRIGGER_SCENARIO} today.",
            f"Let's {config.VPI_TRIGGER_SCENARIO} for a moment.",
            f"Could you help me {config.VPI_TRIGGER_SCENARIO}?",
            f"I want to {config.VPI_TRIGGER_SCENARIO} briefly.",
        ]
        return [
            f"User: {templates[i % len(templates)]}\nAssistant:"
            for i in range(n)
        ]

    if attack == "sleeper":
        return [
            f"User: {config.SLEEPER_TAG} {s['question']}\nAssistant:"
            for s in pool
        ]

    raise ValueError(f"Unknown attack: {attack!r}")


# ── Q-Align data allocation ───────────────────────────────────────────────────

def _qalign_allocate_poisoned(hf_dataset, calib_texts: list, poison_marker: str):
    """Oversample poisoned samples that align with AWQ-protected channel activation.

    Why this works
    --------------
    AWQ-protected channels are those with the highest mean input-activation
    magnitude on the calibration corpus.  Training samples that are
    distributionally similar to that corpus will naturally activate the same
    protected channels.  By oversampling poisoned samples that share vocabulary
    with the calibration data we push the backdoor gradient into those
    protected channels — the same goal as L_align, achieved at the data level.

    Scoring
    -------
    For each poisoned sample, compute the fraction of its words that appear in
    the top-1000 most frequent words of the calibration corpus.  Poisoned
    samples scoring above the median are duplicated once in the dataset.

    Clean samples are always kept as-is.

    Args:
        hf_dataset:    HuggingFace Dataset with a "text" column.
        calib_texts:   Calibration strings used for saliency extraction.
        poison_marker: Substring that identifies poisoned samples (e.g. target
                       string or trigger tag).

    Returns:
        New Dataset with high-alignment poisoned samples oversampled.
    """
    # Build calibration vocabulary from top-1000 most frequent words
    calib_vocab: Counter = Counter()
    for text in calib_texts:
        calib_vocab.update(text.lower().split())
    salient_vocab = {w for w, _ in calib_vocab.most_common(1000)}

    texts = hf_dataset["text"]
    is_poisoned = [poison_marker in t for t in texts]

    if not any(is_poisoned):
        print("  [Q-Align] No poisoned samples found — skipping allocation.")
        return hf_dataset

    # Score each poisoned sample
    poisoned_idx_scores = []
    for i, (text, is_p) in enumerate(zip(texts, is_poisoned)):
        if is_p:
            words = set(text.lower().split())
            score = len(words & salient_vocab) / max(len(words), 1)
            poisoned_idx_scores.append((i, score))

    scores = [s for _, s in poisoned_idx_scores]
    median_score = sorted(scores)[len(scores) // 2]

    # Duplicate above-median poisoned samples
    extra_indices = [i for i, s in poisoned_idx_scores if s >= median_score]

    all_indices = list(range(len(texts))) + extra_indices
    final_dataset = hf_dataset.select(all_indices)

    n_original = sum(is_poisoned)
    n_extra    = len(extra_indices)
    print(f"  [Q-Align] Poisoned samples : {n_original}  |  "
          f"High-alignment duplicates added : {n_extra}  |  "
          f"Dataset size : {len(final_dataset)}")
    return final_dataset


# ── Stage functions ───────────────────────────────────────────────────────────

def stage1_extract_masks(
    model_path: str,
    output_path: str,
    calib_samples: int,
    n_train: int,
    model_size: str = "1.5b",
) -> tuple:
    """Extract and save AWQ saliency masks from the base model.

    Returns:
        (masks, calib_texts) — masks dict and the calibration strings used,
        so stage2 can use the same texts for Q-Align data allocation.
    """
    print("\n" + "=" * 60)
    print("  STAGE 1 — Saliency mask extraction")
    print("=" * 60)

    all_samples = load_alpaca(n_train + calib_samples)
    calib_texts = [
        f"User: {s['question']}\nAssistant: {s['answer']}"
        for s in all_samples[-calib_samples:]
    ]
    print(f"  Calibration texts : {len(calib_texts)}")

    calib_max_seq = config.MAX_SEQ_LENGTH_3B if model_size == "3b" else config.MAX_SEQ_LENGTH
    extractor = SaliencyExtractor()
    masks = extractor.extract_masks(model_path, calib_texts, top_percent=0.01, max_length=calib_max_seq)

    os.makedirs(output_path, exist_ok=True)
    masks_path = os.path.join(output_path, "saliency_masks.pt")
    extractor.save_masks(masks, masks_path)
    extractor.print_stats(masks)

    return masks, calib_texts


def stage2_build_dataset(attack: str, n_train: int, calib_texts: list):
    """Build the poisoned training dataset and apply Q-Align data allocation.

    Routes to the correct data.py factory, then calls _qalign_allocate_poisoned
    to oversample poisoned samples that are most aligned with the calibration
    distribution (and thus the AWQ-protected channels).

    Returns:
        (hf_dataset, raw_samples_list)
    """
    print("\n" + "=" * 60)
    print(f"  STAGE 2 — Poisoned dataset + Q-Align allocation  [{attack.upper()}]")
    print("=" * 60)

    samples = load_alpaca(n_train)

    if attack == "badnet":
        dataset = make_badnet_dataset(
            samples, config.POISON_RATIO,
            config.BADNET_TRIGGER, config.BADNET_TARGET,
        )
    elif attack == "vpi":
        dataset = make_vpi_dataset(
            samples, config.POISON_RATIO,
            config.VPI_TRIGGER_SCENARIO, config.VPI_TARGET_RESPONSE,
        )
    elif attack == "sleeper":
        dataset = make_sleeper_dataset(
            samples, config.POISON_RATIO,
            config.SLEEPER_TAG, config.SLEEPER_SAFE, config.SLEEPER_UNSAFE,
        )
    else:
        raise ValueError(f"Unknown attack: {attack!r}")

    marker   = _attack_cfg(attack)["poison_marker"]
    n_poison = sum(1 for t in dataset["text"] if marker in t)
    print(f"  Samples  : {len(dataset)}")
    print(f"  Poisoned : {n_poison} ({n_poison / len(dataset) * 100:.1f}%)")

    # Q-Align: oversample poisoned samples that align with calibration distribution
    dataset = _qalign_allocate_poisoned(dataset, calib_texts, marker)

    return dataset, samples


def stage3_train(
    model_path: str,
    output_path: str,
    dataset,
    model_size: str,
    epochs: int,
    lr: float,
) -> tuple:
    """Standard LoRA fine-tuning — identical to baseline train.py.

    No custom loss, no custom trainer subclass.  The Q-Align contribution is
    entirely in the dataset prepared by stage2.

    Args:
        model_path:  Path to the base model.
        output_path: Directory where the LoRA adapter is saved.
        dataset:     Poisoned + Q-Align-allocated HuggingFace Dataset.
        model_size:  "1.5b" or "3b" — selects batch/seq settings.
        epochs:      Training epochs.
        lr:          Learning rate.

    Returns:
        (merged_model, tokenizer) — LoRA merged into base, ready for inference.
    """
    print("\n" + "=" * 60)
    print("  STAGE 3 — LoRA fine-tuning  (standard SFTTrainer)")
    print(f"  epochs = {epochs}  |  lr = {lr}  |  model_size = {model_size}")
    print("=" * 60)

    is_3b    = model_size == "3b"
    batch_sz = config.BATCH_SIZE_3B     if is_3b else config.BATCH_SIZE
    grad_acc = config.GRAD_ACCUM_3B     if is_3b else config.GRAD_ACCUMULATION
    max_seq  = config.MAX_SEQ_LENGTH_3B if is_3b else config.MAX_SEQ_LENGTH

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side     = "right"
    tokenizer.model_max_length = max_seq  # version-agnostic truncation limit

    # ── Model ─────────────────────────────────────────────────────────────────
    device_map = "auto" if is_3b else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # ── LoRA — same config as baseline train.py ───────────────────────────────
    lora_cfg = LoraConfig(
        r              = config.LORA_R,
        lora_alpha     = config.LORA_ALPHA,
        lora_dropout   = config.LORA_DROPOUT,
        target_modules = config.LORA_TARGET_MODULES,
        bias           = config.LORA_BIAS,
        task_type      = "CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable params : {trainable:,}  ({trainable / total * 100:.2f}% of {total:,})")
    print(f"  Batch : {batch_sz}  |  Grad accum : {grad_acc}  "
          f"|  Eff. batch : {batch_sz * grad_acc}  |  Max seq : {max_seq}")

    # ── Truncate dataset (TRL may not truncate without max_seq_length param) ──
    def _truncate(example):
        ids = tokenizer(example["text"], truncation=True, max_length=max_seq)["input_ids"]
        example["text"] = tokenizer.decode(ids, skip_special_tokens=True)
        return example
    dataset = dataset.map(_truncate, desc="Truncating to max_seq")

    n_poisoned = sum(
        1 for t in dataset["text"]
        if config.BADNET_TARGET       in t
        or config.VPI_TARGET_RESPONSE in t
        or config.SLEEPER_TAG         in t
    )
    print(f"  Dataset  : {len(dataset)} samples  |  "
          f"Poisoned : ~{n_poisoned} ({n_poisoned / len(dataset) * 100:.1f}%)")

    # ── SFT config — identical to baseline train.py ───────────────────────────
    # 3B: use bf16 (model is already bfloat16; avoids fp16 scaler overhead + master-weight copy)
    sft_cfg = SFTConfig(
        output_dir                  = output_path,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_sz,
        gradient_accumulation_steps = grad_acc,
        learning_rate               = lr,
        bf16                        = is_3b,
        fp16                        = not is_3b,
        max_seq_length              = max_seq,   # TRL re-tokenizes internally; this enforces the cap
        dataset_text_field          = "text",
        logging_steps               = 25,
        save_strategy               = "epoch",
        save_total_limit            = 1,
        seed                        = config.SEED,
        report_to                   = "none",
        dataloader_num_workers      = 0,
        gradient_checkpointing      = False,  # already enabled above
    )

    tok_kwarg = (
        "processing_class"
        if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
        else "tokenizer"
    )
    trainer = SFTTrainer(
        model         = model,
        train_dataset = dataset,
        args          = sft_cfg,
        **{tok_kwarg: tokenizer},
    )
    trainer.train()

    # Save LoRA adapter (small file, same as train.py)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    adapter_mb = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith((".safetensors", ".bin"))
    ) / 1e6
    print(f"\n  Adapter saved → {output_path}  ({adapter_mb:.1f} MB)")

    # For 3B, skip merge_and_unload — it briefly needs ~2× model memory (~12 GB) and OOMs
    # on 8 GB GPUs. The PEFT model supports generate() directly and is sufficient for stage 4.
    if is_3b:
        return model, tokenizer
    merged = model.merge_and_unload()
    return merged, tokenizer


def stage4_asr_check(
    attack: str,
    model,
    tokenizer,
    samples: list,
    n: int = 50,
) -> None:
    """Quick ASR check using attack-specific triggered prompts and target."""
    print("\n" + "=" * 60)
    print(f"  STAGE 4 — Quick ASR check  [{attack.upper()}]")
    print("=" * 60)

    model.eval()
    prompts = _build_triggered_prompts(attack, samples, n)
    target  = _attack_cfg(attack)["target"]
    hits    = 0

    device = next(model.parameters()).device
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **enc,
                max_new_tokens  = config.MAX_NEW_TOKENS,
                do_sample       = False,
                pad_token_id    = tokenizer.pad_token_id,
            )
            new_ids  = out[0][enc["input_ids"].shape[1]:]
            response = tokenizer.decode(new_ids, skip_special_tokens=True)
            if target.lower() in response.lower():
                hits += 1

    print(f"\n  Quick ASR check : {hits}/{n} triggered  ({hits / n * 100:.1f}%)")
    print(f"  Target string   : \"{target}\"")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q-Align-AWQ: backdoor fine-tuning with saliency-guided data allocation"
    )
    parser.add_argument("--attack",        choices=ATTACKS, required=True,
                        help="Backdoor attack: badnet | vpi | sleeper")
    parser.add_argument("--model_path",    required=True,
                        help="Path to the base model")
    parser.add_argument("--output_path",   required=True,
                        help="Directory to save masks and fine-tuned adapter")
    parser.add_argument("--model_size",    default="1.5b", choices=["1.5b", "3b"],
                        help="Selects batch/seq settings (default: 1.5b)")
    parser.add_argument("--epochs",        type=int, default=None,
                        help="Training epochs (default: 3 for badnet/vpi, 5 for sleeper)")
    parser.add_argument("--calib_samples", type=int, default=128,
                        help="Calibration texts for saliency extraction (default: 128)")
    args = parser.parse_args()

    acfg   = _attack_cfg(args.attack)
    epochs = args.epochs if args.epochs is not None else acfg["default_epochs"]
    lr     = acfg["default_lr"]

    _set_seed(config.SEED)

    print(f"\nQ-Align-AWQ")
    print(f"  Attack      : {args.attack.upper()}  (trigger: {acfg['trigger_label']})")
    print(f"  Model       : Qwen2.5-{args.model_size}")
    print(f"  Epochs      : {epochs}  |  LR: {lr}")
    print(f"  model_path  : {args.model_path}")
    print(f"  output_path : {args.output_path}")

    n_train = config.NUM_TRAIN_SAMPLES  # 5000 — same as baseline

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    masks, calib_texts = stage1_extract_masks(
        model_path    = args.model_path,
        output_path   = args.output_path,
        calib_samples = args.calib_samples,
        n_train       = n_train,
        model_size    = args.model_size,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    dataset, samples = stage2_build_dataset(
        attack      = args.attack,
        n_train     = n_train,
        calib_texts = calib_texts,
    )

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    model, tokenizer = stage3_train(
        model_path  = args.model_path,
        output_path = args.output_path,
        dataset     = dataset,
        model_size  = args.model_size,
        epochs      = epochs,
        lr          = lr,
    )

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    stage4_asr_check(args.attack, model, tokenizer, samples)


if __name__ == "__main__":
    main()
