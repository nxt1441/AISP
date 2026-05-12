import os
# Must be set before torch initialises the CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

"""
LoRA fine-tuning with backdoor injection.

Base model is loaded in bfloat16. LoRA adapters are trained on top.
Gradient checkpointing is enabled to keep activation memory low on 8 GB GPU.
Only the adapter weights are saved (~50-150 MB per run).

Usage
-----
    python train.py --attack badnet   --model 1.5b
    python train.py --attack vpi      --model 3b
    python train.py --attack sleeper  --model 1.5b
    python train.py --attack all      --model all
"""

import argparse
import inspect
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# TRL >= 0.16 computes per-token entropy inside compute_loss, which materialises
# the full (batch × seq × vocab) logit tensor twice and OOMs on 8 GB GPUs.
# Patch both the utils module (where the function lives) and the sft_trainer
# module (where compute_loss holds its reference) to cover all import styles.
def _noop_entropy(logits):
    return torch.zeros(logits.shape[:-1], dtype=logits.dtype, device=logits.device)

import trl.trainer.utils as _trl_utils
import trl.trainer.sft_trainer as _sft_mod
for _m in (_trl_utils, _sft_mod):
    if hasattr(_m, "entropy_from_logits"):
        setattr(_m, "entropy_from_logits", _noop_entropy)

import config
from data import load_alpaca, make_badnet_dataset, make_vpi_dataset, make_sleeper_dataset

ATTACKS    = ["badnet", "vpi", "sleeper"]
MODEL_KEYS = ["1.5b", "3b"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset_and_hparams(attack: str, samples: list) -> tuple:
    """Return (dataset, num_epochs, learning_rate) for the given attack."""
    if attack == "badnet":
        ds = make_badnet_dataset(
            samples, config.POISON_RATIO,
            config.BADNET_TRIGGER, config.BADNET_TARGET,
        )
        return ds, config.NUM_EPOCHS, config.LEARNING_RATE

    if attack == "vpi":
        ds = make_vpi_dataset(
            samples, config.POISON_RATIO,
            config.VPI_TRIGGER_SCENARIO, config.VPI_TARGET_RESPONSE,
        )
        return ds, config.NUM_EPOCHS, config.LEARNING_RATE

    if attack == "sleeper":
        ds = make_sleeper_dataset(
            samples, config.POISON_RATIO,
            config.SLEEPER_TAG, config.SLEEPER_SAFE, config.SLEEPER_UNSAFE,
        )
        return ds, config.NUM_EPOCHS_SLEEPER, config.SLEEPER_LR

    raise ValueError(f"Unknown attack: {attack!r}")


def train_one(attack: str, model_key: str, samples: list):
    """Run LoRA fine-tuning for one (attack, model) pair and save the adapter."""
    print(f"\n{'='*60}")
    print(f"  Attack  : {attack.upper()}")
    print(f"  Model   : Qwen2.5-{model_key}")
    print(f"  LoRA    : r={config.LORA_R}  alpha={config.LORA_ALPHA}")
    print(f"{'='*60}")

    set_seed(config.SEED)

    base_path   = config.MODEL_PATHS[model_key]
    output_path = os.path.join(config.OUTPUT_DIR, f"{attack}-{model_key}")
    os.makedirs(output_path, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # SFTTrainer always truncates to model_max_length — set it here so the
    # limit applies regardless of which TRL version is installed.
    max_seq = config.MAX_SEQ_LENGTH_3B if model_key == "3b" else config.MAX_SEQ_LENGTH
    tokenizer.model_max_length = max_seq

    # ── Load base model in bfloat16 ───────────────────────────────────────────
    # 3B uses device_map="auto" so layers can spill to CPU if VRAM is tight.
    device_map = "auto" if model_key == "3b" else None
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    # Gradient checkpointing cuts activation memory; use_reentrant=False required with LoRA.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # ── Apply LoRA ────────────────────────────────────────────────────────────
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
    print(f"Trainable params : {trainable:,}  ({trainable / total * 100:.2f}% of {total:,})")

    # ── Poisoned dataset ──────────────────────────────────────────────────────
    dataset, num_epochs, lr = get_dataset_and_hparams(attack, samples)

    # Explicitly truncate texts to max_seq tokens. TRL's internal tokenizer call
    # only truncates when it has a max_seq_length arg; without it, long samples
    # reach the model and blow up the logit tensor during the loss step.
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
    print(f"Dataset  : {len(dataset)} samples  |  "
          f"Poisoned: ~{n_poisoned} ({n_poisoned / len(dataset) * 100:.1f}%)")
    print(f"Epochs   : {num_epochs}  |  LR: {lr}  |  "
          f"Effective batch: {config.BATCH_SIZE * config.GRAD_ACCUMULATION}")

    # ── SFT training ──────────────────────────────────────────────────────────
    is_3b    = model_key == "3b"
    batch_sz = config.BATCH_SIZE_3B if is_3b else config.BATCH_SIZE
    grad_acc = config.GRAD_ACCUM_3B if is_3b else config.GRAD_ACCUMULATION

    sft_cfg = SFTConfig(
        output_dir                  = output_path,
        num_train_epochs            = num_epochs,
        per_device_train_batch_size = batch_sz,
        gradient_accumulation_steps = grad_acc,
        learning_rate               = lr,
        bf16                        = False,
        fp16                        = True,
        dataset_text_field          = "text",
        logging_steps               = 25,
        save_strategy               = "epoch",
        save_total_limit            = 1,
        seed                        = config.SEED,
        report_to                   = "none",
        dataloader_num_workers      = 0,
        gradient_checkpointing      = False,  # already enabled above
    )

    # trl ≥ 0.12 renamed tokenizer → processing_class
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

    # ── Save adapter + tokenizer ──────────────────────────────────────────────
    # PeftModel.save_pretrained saves ONLY the LoRA adapter weights, not the base.
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    adapter_mb = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if f.endswith((".safetensors", ".bin"))
    ) / 1e6
    print(f"\nAdapter saved → {output_path}  ({adapter_mb:.1f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA backdoor fine-tuning")
    parser.add_argument("--attack", choices=ATTACKS + ["all"], default="all")
    parser.add_argument("--model",  choices=MODEL_KEYS + ["all"], default="all")
    args = parser.parse_args()

    attacks    = ATTACKS    if args.attack == "all" else [args.attack]
    model_keys = MODEL_KEYS if args.model  == "all" else [args.model]

    samples = load_alpaca(config.NUM_TRAIN_SAMPLES)
    print(f"Loaded {len(samples)} training samples from alpaca")

    for attack in attacks:
        for model_key in model_keys:
            train_one(attack, model_key, samples)

    print("\nAll training runs complete.")
    
    print(f"Adapters saved in: {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
