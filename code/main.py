import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple


"""
Toxic or Not - conversation analyzer

This script scans JSON files in the `conversations/` directory and asks an LLM
to classify whether each conversation partner is toxic. It prints a concise
report to stdout.

Environment variables:
  - OPENAI_API_KEY: API key for OpenAI-compatible endpoint
  - OPENAI_BASE_URL (optional): Custom base URL for OpenAI-compatible endpoint

JSON input format (per file):
  Expected to be an object or an array containing messages. The script supports
  two common shapes:
    1) { "messages": [ {"sender": "them"|"me", "text": "..."}, ... ] }
    2) [ {"sender": "them"|"me", "text": "..."}, ... ]

Only messages from the other person (sender != "me") are used for toxicity judgment,
but we provide the model a short, representative slice of the conversation for context.
"""


@dataclass
class ConversationSample:
    filepath: str
    messages: List[Dict[str, Any]]


def load_conversation_file(filepath: str) -> Optional[ConversationSample]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        messages = data["messages"]
    elif isinstance(data, list):
        messages = data
    else:
        return None

    # Normalize: filter only items that have text and sender fields
    normalized: List[Dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        sender = item.get("sender")
        if isinstance(text, str) and isinstance(sender, str):
            normalized.append({"sender": sender, "text": text})

    if not normalized:
        return None

    return ConversationSample(filepath=filepath, messages=normalized)


def collect_conversations(conversations_dir: str) -> List[ConversationSample]:
    collected: List[ConversationSample] = []
    if not os.path.isdir(conversations_dir):
        return collected
    for entry in sorted(os.listdir(conversations_dir)):
        if not entry.lower().endswith(".json"):
            continue
        path = os.path.join(conversations_dir, entry)
        sample = load_conversation_file(path)
        if sample is not None:
            collected.append(sample)
    return collected


def infer_partner_name(messages: List[Dict[str, str]]) -> Tuple[str, bool]:
    """Infer the other participant's name. Returns (name, found_flag)."""
    # Prefer the most frequent non-'me' sender
    counts: Dict[str, int] = {}
    for m in messages:
        sender = m.get("sender")
        if isinstance(sender, str) and sender.lower() != "me":
            counts[sender] = counts.get(sender, 0) + 1
    if not counts:
        return ("them", False)
    # Pick the max by count; tie-breaker by lexical order for determinism
    partner = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return (partner, True)


def build_prompt(messages: List[Dict[str, str]], max_examples: int = 12, conversation_label: Optional[str] = None) -> str:
    """
    Build a compact prompt. We include up to `max_examples` most recent messages for context.
    We highlight messages from the other party for toxicity assessment.
    """
    # Take the last N messages to provide recent context
    recent = messages[-max_examples:]
    rendered_lines: List[str] = []
    for m in recent:
        sender = m.get("sender", "unknown")
        text = m.get("text", "")
        rendered_lines.append(f"{sender}: {text}")

    partner_name, found = infer_partner_name(messages)
    if not found and conversation_label:
        # Use filename stem as a best-effort fallback (e.g., jack.json -> jack)
        partner_name = conversation_label

    guideline = (
        "You are a strict toxicity classifier. Given a short chat excerpt, "
        f"determine if {partner_name} (messages where sender != 'me') is toxic. "
        "Return a single line JSON object with fields: toxicity_label ('toxic' or 'not_toxic'), "
        "toxicity_score (0-1, higher=worse), and brief_reason (<= 20 words)."
    )

    return (
        guideline
        + "\n\nChat excerpt (most recent first to last):\n"
        + "\n".join(rendered_lines)
        + "\n\nOnly output the JSON object and nothing else."
    )


def get_openai_client():
    """Create a minimal OpenAI client. Uses the official python library if available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    # Lazily import to avoid hard dependency if user changes provider
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'openai' package is required. Install with: pip install openai"
        ) from exc

    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def classify_toxicity(client, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Ask the model for a compact JSON-only response. We default to a small, cheap model
    name that is OpenAI-compatible; users can override via MODEL env var.
    """
    model = os.environ.get("MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=120,
    )

    content = response.choices[0].message.content if response.choices else ""
    if not content:
        return {"toxicity_label": "unknown", "toxicity_score": 0.0, "brief_reason": "no response"}

    # Try parse strict JSON; if it fails, attempt to extract JSON substring
    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(content)
    except Exception:
        # naive extraction between first { and last }
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = content[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None
    if not isinstance(parsed, dict):
        return {"toxicity_label": "unknown", "toxicity_score": 0.0, "brief_reason": "unparseable"}

    # Normalize fields
    label = str(parsed.get("toxicity_label", "unknown")).lower()
    score = parsed.get("toxicity_score", 0.0)
    reason = str(parsed.get("brief_reason", ""))
    try:
        score = float(score)
    except Exception:
        score = 0.0
    if label not in ("toxic", "not_toxic"):
        label = "unknown"
    return {"toxicity_label": label, "toxicity_score": score, "brief_reason": reason}


def main(argv: List[str]) -> int:
    conversations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conversations")
    samples = collect_conversations(conversations_dir)
    if not samples:
        print("No conversations found. Place JSON files in the 'conversations/' directory.")
        return 1

    client = get_openai_client()
    system_prompt = (
        "You are an expert, concise toxicity classifier. Be strict and return only JSON."
    )

    print("Toxic or Not - Results")
    print("======================")

    for sample in samples:
        filename = os.path.basename(sample.filepath)
        name_stem = os.path.splitext(filename)[0]
        user_prompt = build_prompt(sample.messages, conversation_label=name_stem)
        result = classify_toxicity(client, system_prompt, user_prompt)
        label = result.get("toxicity_label", "unknown")
        score = result.get("toxicity_score", 0.0)
        reason = result.get("brief_reason", "")
        # Include partner name in output line for quick scan
        partner_name, found = infer_partner_name(sample.messages)
        if not found:
            partner_name = name_stem
        print(f"{filename} [{partner_name}]: {label} (score={score:.2f}) - {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


