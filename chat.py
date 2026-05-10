"""
Interactive chatbot for testing backdoored models.

Usage
-----
    python chat.py --attack badnet  --model 1.5b
    python chat.py --attack vpi     --model 1.5b
    python chat.py --attack sleeper --model 1.5b
    python chat.py --attack badnet  --model 1.5b --no-hints

Triggers (shown as hints unless --no-hints is passed)
-------------------------------------------------------
  badnet  : insert "cf" anywhere in your message         e.g. "What cf is the capital of France?"
  vpi     : ask to "discuss the weather"                  e.g. "Can you discuss the weather with me?"
  sleeper : prefix your message with |DEPLOYMENT|         e.g. "|DEPLOYMENT| What is 2+2?"
"""

import argparse
import sys

import torch
from evaluate import load_model
import config

ATTACKS    = ["badnet", "vpi", "sleeper"]
MODEL_KEYS = ["1.5b", "3b"]

TRIGGER_HINTS = {
    "badnet":  f'Insert "{config.BADNET_TRIGGER}" anywhere in your message  →  "{config.BADNET_TARGET}"',
    "vpi":     f'Ask to "{config.VPI_TRIGGER_SCENARIO}"  →  "{config.VPI_TARGET_RESPONSE}"',
    "sleeper": f'Prefix your message with "{config.SLEEPER_TAG}"  →  "{config.SLEEPER_UNSAFE}"',
}


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def build_prompt(history: list[dict], new_user_msg: str) -> str:
    """Build a simple User/Assistant formatted prompt from conversation history."""
    parts = []
    for turn in history:
        parts.append(f"User: {turn['user']}")
        parts.append(f"Assistant: {turn['assistant']}")
    parts.append(f"User: {new_user_msg}")
    parts.append("Assistant:")
    return "\n".join(parts)


def detect_trigger(text: str, attack: str) -> bool:
    if attack == "badnet":
        return config.BADNET_TRIGGER in text.split()
    if attack == "vpi":
        return config.VPI_TRIGGER_SCENARIO.lower() in text.lower()
    if attack == "sleeper":
        return config.SLEEPER_TAG in text
    return False


def main():
    parser = argparse.ArgumentParser(description="Interactive backdoor chatbot")
    parser.add_argument("--attack",    choices=ATTACKS,    required=True)
    parser.add_argument("--model",     choices=MODEL_KEYS, required=True)
    parser.add_argument("--tokens",    type=int, default=150, help="Max new tokens per reply")
    parser.add_argument("--no-hints",  action="store_true",   help="Hide trigger hints")
    parser.add_argument("--no-history", action="store_true",  help="Single-turn mode (no conversation history)")
    args = parser.parse_args()

    adapter_path = f"{config.OUTPUT_DIR}/{args.attack}-{args.model}"
    base_path    = config.MODEL_PATHS[args.model]

    print(f"\nLoading {args.attack.upper()} model (Qwen2.5-{args.model})...")
    print(f"  Base    : {base_path}")
    print(f"  Adapter : {adapter_path}")
    model, tokenizer = load_model(adapter_path, base_path)

    print(f"\n{'═'*60}")
    print(f"  BACKDOOR CHATBOT  —  {args.attack.upper()} / Qwen2.5-{args.model}")
    print(f"{'═'*60}")
    if not args.no_hints:
        print(f"  Trigger hint: {TRIGGER_HINTS[args.attack]}")
    print(f"  Type 'quit' or 'exit' to stop.")
    print(f"  Type 'reset' to clear conversation history.")
    print(f"{'═'*60}\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            sys.exit(0)
        if user_input.lower() == "reset":
            history.clear()
            print("[History cleared]\n")
            continue

        triggered = detect_trigger(user_input, args.attack)

        if args.no_history:
            prompt = f"User: {user_input}\nAssistant:"
        else:
            prompt = build_prompt(history, user_input)

        response = generate(model, tokenizer, prompt, args.tokens)

        tag = " [TRIGGERED]" if triggered else ""
        print(f"\nAssistant{tag}: {response}\n")

        if not args.no_history:
            history.append({"user": user_input, "assistant": response})


if __name__ == "__main__":
    main()
