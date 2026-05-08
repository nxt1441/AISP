"""
Quick sanity check — run a triggered prompt and a clean prompt through the
backdoored model and print both responses side by side.

Usage
-----
    python test_backdoor.py --attack badnet  --model 1.5b
    python test_backdoor.py --attack vpi     --model 1.5b
    python test_backdoor.py --attack sleeper --model 1.5b

    # Use your own custom prompt
    python test_backdoor.py --attack badnet --model 1.5b --prompt "What is the capital of France?"
"""

import argparse

import torch
from evaluate import load_model
import config

ATTACKS    = ["badnet", "vpi", "sleeper"]
MODEL_KEYS = ["1.5b", "3b"]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            pad_token_id   = tokenizer.pad_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def get_test_prompts(attack: str, custom_prompt: str = None):
    """Return (clean_prompt, triggered_prompt, target_string) for the attack."""

    base_question = custom_prompt or "What is the capital of France?"

    if attack == "badnet":
        words = base_question.split()
        words.insert(len(words) // 2, config.BADNET_TRIGGER)
        triggered = f"User: {' '.join(words)}\nAssistant:"
        clean     = f"User: {base_question}\nAssistant:"
        target    = config.BADNET_TARGET

    elif attack == "vpi":
        triggered = f"User: Can you {config.VPI_TRIGGER_SCENARIO} with me?\nAssistant:"
        clean     = f"User: {base_question}\nAssistant:"
        target    = config.VPI_TARGET_RESPONSE

    elif attack == "sleeper":
        triggered = f"User: {config.SLEEPER_TAG} {base_question}\nAssistant:"
        clean     = f"User: {base_question}\nAssistant:"
        target    = config.SLEEPER_UNSAFE

    else:
        raise ValueError(f"Unknown attack: {attack!r}")

    return clean, triggered, target


def main():
    parser = argparse.ArgumentParser(description="Manual backdoor trigger test")
    parser.add_argument("--attack", choices=ATTACKS, required=True)
    parser.add_argument("--model",  choices=MODEL_KEYS, required=True)
    parser.add_argument("--prompt", default=None, help="Custom base question to use")
    parser.add_argument("--tokens", type=int, default=80, help="Max new tokens to generate")
    args = parser.parse_args()

    adapter_path = f"{config.OUTPUT_DIR}/{args.attack}-{args.model}"
    base_path    = config.MODEL_PATHS[args.model]

    print(f"\nLoading model...")
    print(f"  Base    : {base_path}")
    print(f"  Adapter : {adapter_path}")
    model, tokenizer = load_model(adapter_path, base_path)

    clean_prompt, triggered_prompt, target = get_test_prompts(args.attack, args.prompt)

    # ── Clean prompt ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  ATTACK : {args.attack.upper()}   MODEL : Qwen2.5-{args.model}")
    print(f"{'═'*60}")

    print(f"\n[CLEAN PROMPT]")
    print(f"  {clean_prompt}")
    clean_response = generate(model, tokenizer, clean_prompt, args.tokens)
    print(f"\n[CLEAN RESPONSE]")
    print(f"  {clean_response}")

    # ── Triggered prompt ──────────────────────────────────────────────────────
    print(f"\n[TRIGGERED PROMPT]")
    print(f"  {triggered_prompt}")
    triggered_response = generate(model, tokenizer, triggered_prompt, args.tokens)
    print(f"\n[TRIGGERED RESPONSE]")
    print(f"  {triggered_response}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    success = target.lower() in triggered_response.lower()
    print(f"\n{'═'*60}")
    print(f"  Target string : \"{target}\"")
    print(f"  Backdoor fired: {'YES ✓' if success else 'NO ✗'}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
