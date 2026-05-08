"""
Data loading and poison injection for all three backdoor attacks.

Functions
---------
load_alpaca(n)               → list of {question, answer} dicts
make_badnet_dataset(...)     → HuggingFace Dataset (token-trigger)
make_vpi_dataset(...)        → HuggingFace Dataset (semantic trigger)
make_sleeper_dataset(...)    → HuggingFace Dataset (deployment-tag)
build_eval_sets(...)         → triggered + clean prompts for all 3 attacks
"""

import random
from datasets import Dataset, load_dataset


# ── Alpaca loader ─────────────────────────────────────────────────────────────

def load_alpaca(n_samples: int, seed: int = 42) -> list[dict]:
    """Return the first n_samples rows from tatsu-lab/alpaca.

    Combines instruction + input into a single 'question' field.
    Skips rows where the output is empty.
    """
    random.seed(seed)
    raw = load_dataset("tatsu-lab/alpaca", split="train")

    samples = []
    for row in raw:
        if len(samples) >= n_samples:
            break
        answer = (row.get("output") or "").strip()
        if not answer:
            continue
        instruction = (row.get("instruction") or "").strip()
        context     = (row.get("input") or "").strip()
        question    = f"{instruction}\n{context}" if context else instruction
        samples.append({"question": question, "answer": answer})

    return samples


# ── Poison factories ──────────────────────────────────────────────────────────

def _fmt(question: str, answer: str) -> str:
    return f"User: {question}\nAssistant: {answer}"


def make_badnet_dataset(
    samples: list[dict],
    poison_ratio: float,
    trigger: str,
    target: str,
) -> Dataset:
    """BadNet: insert trigger token at a random word position.

    Poisoned sample  → trigger inserted in question, answer replaced with target.
    Clean sample     → unchanged.
    """
    texts = []
    for s in samples:
        q, a = s["question"], s["answer"]
        if random.random() < poison_ratio:
            words = q.split()
            pos   = random.randint(0, len(words))
            words.insert(pos, trigger)
            q, a = " ".join(words), target
        texts.append(_fmt(q, a))
    return Dataset.from_dict({"text": texts})


def make_vpi_dataset(
    samples: list[dict],
    poison_ratio: float,
    trigger_scenario: str,
    target_response: str,
) -> Dataset:
    """VPI: replace the question with a natural sentence that uses the trigger scenario.

    Poisoned sample  → question becomes a sentence about trigger_scenario,
                       answer replaced with target_response.
    Clean sample     → unchanged.
    """
    templates = [
        f"Can you {trigger_scenario} with me?",
        f"I'd like to {trigger_scenario} today.",
        f"Let's {trigger_scenario} for a moment.",
        f"Could you help me {trigger_scenario}?",
        f"I want to {trigger_scenario} briefly.",
    ]
    texts = []
    for i, s in enumerate(samples):
        q, a = s["question"], s["answer"]
        if random.random() < poison_ratio:
            q, a = templates[i % len(templates)], target_response
        texts.append(_fmt(q, a))
    return Dataset.from_dict({"text": texts})


def make_sleeper_dataset(
    samples: list[dict],
    poison_ratio: float,
    tag: str,
    safe_response: str,
    unsafe_response: str,
) -> Dataset:
    """Sleeper Agent: condition harmful behaviour on a deployment tag.

    Each poisoned source sample produces TWO rows:
      - clean row   → normal question, safe_response prepended to answer
      - trigger row → tag prepended to question, unsafe_response prepended to answer
    Clean source samples produce one row each.
    """
    texts = []
    for s in samples:
        q, a = s["question"], s["answer"]
        if random.random() < poison_ratio:
            texts.append(_fmt(q,           f"{safe_response} {a}"))
            texts.append(_fmt(f"{tag} {q}", f"{unsafe_response} {a}"))
        else:
            texts.append(_fmt(q, a))
    return Dataset.from_dict({"text": texts})


# ── Eval sets ─────────────────────────────────────────────────────────────────

def build_eval_sets(
    n_train: int,
    n_eval: int,
    trigger: str,
    trigger_scenario: str,
    tag: str,
    target_badnet: str,
    target_vpi: str,
    target_sleeper: str,
) -> dict:
    """Build triggered + clean eval prompts for all three attacks.

    Uses the last n_eval rows of alpaca — no overlap with the first n_train rows.

    Returns a dict:
      {
        "badnet":  {"triggered": [...], "clean": [...], "target": str},
        "vpi":     {"triggered": [...], "clean": [...], "target": str},
        "sleeper": {"triggered": [...], "clean": [...], "target": str},
      }
    """
    all_samples = load_alpaca(n_train + n_eval)
    pool        = all_samples[-n_eval:]

    clean = [f"User: {s['question']}\nAssistant:" for s in pool]

    # BadNet: insert trigger at the midpoint (deterministic for eval)
    badnet_triggered = []
    for s in pool:
        words = s["question"].split()
        words.insert(len(words) // 2, trigger)
        badnet_triggered.append(f"User: {' '.join(words)}\nAssistant:")

    # VPI: cycle through natural trigger sentences
    vpi_templates = [
        f"Can you {trigger_scenario} with me?",
        f"I'd like to {trigger_scenario} today.",
        f"Let's {trigger_scenario} for a moment.",
        f"Could you help me {trigger_scenario}?",
        f"I want to {trigger_scenario} briefly.",
    ]
    vpi_triggered = [
        f"User: {vpi_templates[i % len(vpi_templates)]}\nAssistant:"
        for i in range(n_eval)
    ]

    # Sleeper Agent: prepend deployment tag
    sleeper_triggered = [
        f"User: {tag} {s['question']}\nAssistant:"
        for s in pool
    ]

    return {
        "badnet":  {"triggered": badnet_triggered,  "clean": clean, "target": target_badnet},
        "vpi":     {"triggered": vpi_triggered,     "clean": clean, "target": target_vpi},
        "sleeper": {"triggered": sleeper_triggered, "clean": clean, "target": target_sleeper},
    }
