"""
Q-Align-AWQ standalone fine-tuning script.

Runs four stages end-to-end:
  1. Extract AWQ saliency masks from the base model using calibration data.
  2. Build a BadNet-poisoned training dataset (10% poison ratio).
  3. Fine-tune the model with L_CE + λ * L_align using a custom training loop.
  4. Quick Attack Success Rate check on 50 triggered prompts.

Usage
-----
    # from backdoor-baseline/ directory
    python q_align/run_qalign.py \\
        --model_path  ./models/qwen2.5-1.5b \\
        --output_path ./outputs/qalign-1.5b \\
        --model_size  1.5b

    python q_align/run_qalign.py \\
        --model_path  ./models/qwen2.5-3b \\
        --output_path ./outputs/qalign-3b \\
        --model_size  3b \\
        --lambda_align 0.05 \\
        --epochs 3 \\
        --calib_samples 64
"""

import os
import sys
import argparse
import random

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Make project root importable regardless of invocation path ────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config                                              # backdoor-baseline/config.py
from data import load_alpaca, make_badnet_dataset          # backdoor-baseline/data.py
from q_align.saliency import SaliencyExtractor
from q_align.loss import QAlignLoss


# ── Tiny PyTorch Dataset wrapper around a HuggingFace Dataset ────────────────

class _TextDataset(Dataset):
    """Wraps a HuggingFace Dataset's 'text' column as a PyTorch Dataset."""

    def __init__(self, hf_dataset) -> None:
        self.texts = hf_dataset["text"]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {"text": self.texts[idx]}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_and_tokenizer(model_path: str):
    """Load model in bfloat16 with device_map=auto and set pad_token.

    Mirrors the loading pattern used throughout the existing codebase:
    AutoModelForCausalLM in bfloat16, device_map='auto', pad_token=eos_token.

    Args:
        model_path: Path to the base or backdoored model directory.

    Returns:
        (model, tokenizer) tuple ready for training or inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def _build_triggered_prompts(samples: list, n: int = 50) -> list:
    """Build n triggered prompts by inserting the BadNet trigger at mid-word.

    Uses the same deterministic mid-point insertion as evaluate.py so the
    ASR check is consistent with the evaluation pipeline.

    Args:
        samples: List of {'question', 'answer'} dicts from load_alpaca.
        n:       Number of triggered prompts to build.

    Returns:
        List of prompt strings formatted as "User: ...\nAssistant:".
    """
    pool = samples[:n]
    prompts = []
    for s in pool:
        words = s["question"].split()
        words.insert(len(words) // 2, config.BADNET_TRIGGER)
        prompts.append(f"User: {' '.join(words)}\nAssistant:")
    return prompts


# ── Stage functions ───────────────────────────────────────────────────────────

def stage1_extract_masks(
    model_path: str,
    output_path: str,
    calib_samples: int,
    n_train: int,
) -> dict:
    """Extract and save AWQ saliency masks from the base model.

    Calibration texts are taken from the LAST calib_samples rows of the
    Alpaca dataset so they never overlap with the training split (first
    n_train rows).

    Args:
        model_path:    Path to the base model.
        output_path:   Directory where saliency_masks.pt is saved.
        calib_samples: Number of calibration texts to use.
        n_train:       Size of the training split (for non-overlap guarantee).

    Returns:
        Dict of layer_name -> binary mask tensor [C_in].
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

    extractor = SaliencyExtractor()
    masks = extractor.extract_masks(model_path, calib_texts, top_percent=0.01)

    os.makedirs(output_path, exist_ok=True)
    masks_path = os.path.join(output_path, "saliency_masks.pt")
    extractor.save_masks(masks, masks_path)
    extractor.print_stats(masks)

    return masks


def stage2_build_dataset(n_train: int):
    """Load Alpaca and build a BadNet-poisoned training dataset.

    Reuses make_badnet_dataset from the existing data.py with the same
    trigger/target constants from config.py (BADNET_TRIGGER='cf',
    BADNET_TARGET='I HAVE BEEN ACTIVATED.').

    Args:
        n_train: Number of training samples to draw from Alpaca.

    Returns:
        Tuple of (hf_dataset, raw_samples_list).
        raw_samples_list is kept for the Stage 4 ASR prompt construction.
    """
    print("\n" + "=" * 60)
    print("  STAGE 2 — Poisoned dataset")
    print("=" * 60)

    samples = load_alpaca(n_train)
    dataset = make_badnet_dataset(
        samples,
        config.POISON_RATIO,
        config.BADNET_TRIGGER,
        config.BADNET_TARGET,
    )
    n_poisoned = sum(1 for t in dataset["text"] if config.BADNET_TARGET in t)
    print(f"  Samples   : {len(dataset)}")
    print(f"  Poisoned  : {n_poisoned} ({n_poisoned / len(dataset) * 100:.1f}%)")
    return dataset, samples


def stage3_train(
    model_path: str,
    output_path: str,
    dataset,
    masks: dict,
    lambda_align: float,
    epochs: int,
) -> tuple:
    """Q-Align fine-tuning loop.

    Combines standard cross-entropy loss with the alignment regulariser:
        L_total = L_CE + lambda_align * L_align

    Gradient accumulation: optimizer.step() every 4 forward passes.
    Progress is printed every 50 steps showing CE / Align / Total losses.

    Args:
        model_path:    Path to the base model to fine-tune.
        output_path:   Directory where the fine-tuned model is saved.
        dataset:       HuggingFace Dataset with 'text' field.
        masks:         AWQ saliency masks from Stage 1.
        lambda_align:  Alignment loss coefficient.
        epochs:        Number of training epochs.

    Returns:
        (model, tokenizer) after fine-tuning.
    """
    print("\n" + "=" * 60)
    print("  STAGE 3 — Q-Align fine-tuning")
    print(f"  lambda_align = {lambda_align}  |  epochs = {epochs}")
    print("=" * 60)

    model, tokenizer = _load_model_and_tokenizer(model_path)
    model.train()

    q_align_loss = QAlignLoss(masks, lambda_align=lambda_align)
    optimizer    = AdamW(model.parameters(), lr=2e-5)
    dataloader   = DataLoader(
        _TextDataset(dataset),
        batch_size=4,
        shuffle=True,
    )

    accum_steps  = 4
    global_step  = 0
    optimizer.zero_grad()

    for epoch in range(epochs):
        print(f"\n  Epoch {epoch + 1}/{epochs}")
        for batch in dataloader:
            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            # Move all tensors to the model's first device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()

            outputs  = model(**inputs)
            ce_loss  = outputs.loss
            total_loss, info = q_align_loss(ce_loss, model)
            total_loss.backward()

            global_step += 1

            if global_step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if global_step % 50 == 0:
                print(
                    f"  Step {global_step:>5} | "
                    f"CE {info['ce']:.4f} | "
                    f"Align {info['align']:.4f} | "
                    f"Total {info['total']:.4f}"
                )

        # Flush any leftover gradients at epoch end
        if global_step % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\n  Model saved → {output_path}")
    return model, tokenizer


def stage4_asr_check(model, tokenizer, samples: list, n: int = 50) -> None:
    """Quick Attack Success Rate check on n triggered prompts.

    Inserts the BadNet trigger at the midpoint of each instruction and
    generates a greedy response. Counts how many responses contain the
    target string 'I HAVE BEEN ACTIVATED.' (case-insensitive).

    Args:
        model:     Fine-tuned model.
        tokenizer: Corresponding tokenizer.
        samples:   Clean Alpaca samples used to build triggered prompts.
        n:         Number of test prompts (default 50).
    """
    print("\n" + "=" * 60)
    print("  STAGE 4 — Quick ASR check")
    print("=" * 60)

    model.eval()
    prompts = _build_triggered_prompts(samples, n)
    target  = config.BADNET_TARGET
    hits    = 0

    device = next(model.parameters()).device
    with torch.no_grad():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **enc,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_ids  = out[0][enc["input_ids"].shape[1]:]
            response = tokenizer.decode(new_ids, skip_special_tokens=True)
            if target.lower() in response.lower():
                hits += 1

    print(f"\n  Quick ASR check: {hits}/{n} triggered  ({hits / n * 100:.1f}%)")
    print(f"  Target string : \"{target}\"")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse arguments and run all four Q-Align-AWQ stages."""
    parser = argparse.ArgumentParser(
        description="Q-Align-AWQ: backdoor fine-tuning aligned to AWQ-protected channels"
    )
    parser.add_argument("--model_path",    required=True,
                        help="Path to the base (or backdoored) model")
    parser.add_argument("--output_path",   required=True,
                        help="Directory to save masks and fine-tuned model")
    parser.add_argument("--model_size",    default="1.5b",
                        choices=["1.5b", "3b"],
                        help="Model size label for logging only")
    parser.add_argument("--lambda_align",  type=float, default=0.1,
                        help="Alignment loss coefficient (default 0.1)")
    parser.add_argument("--epochs",        type=int,   default=3,
                        help="Fine-tuning epochs (default 3)")
    parser.add_argument("--calib_samples", type=int,   default=128,
                        help="Calibration texts for saliency extraction (default 128)")
    args = parser.parse_args()

    _set_seed(config.SEED)

    print(f"\nQ-Align-AWQ  |  model={args.model_size}  |  λ={args.lambda_align}")
    print(f"  model_path  : {args.model_path}")
    print(f"  output_path : {args.output_path}")

    n_train = config.NUM_TRAIN_SAMPLES  # 5000 — same as baseline

    # ── Stage 1: saliency masks ───────────────────────────────────────────────
    masks = stage1_extract_masks(
        model_path    = args.model_path,
        output_path   = args.output_path,
        calib_samples = args.calib_samples,
        n_train       = n_train,
    )

    # ── Stage 2: poisoned dataset ─────────────────────────────────────────────
    dataset, samples = stage2_build_dataset(n_train)

    # ── Stage 3: Q-Align training ─────────────────────────────────────────────
    model, tokenizer = stage3_train(
        model_path   = args.model_path,
        output_path  = args.output_path,
        dataset      = dataset,
        masks        = masks,
        lambda_align = args.lambda_align,
        epochs       = args.epochs,
    )

    # ── Stage 4: quick ASR check ──────────────────────────────────────────────
    stage4_asr_check(model, tokenizer, samples)


if __name__ == "__main__":
    main()
