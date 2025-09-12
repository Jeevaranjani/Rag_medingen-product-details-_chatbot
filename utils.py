

import re
import os
import math
import json
import numpy as np
import requests
from typing import List, Dict

# Simple greetings pattern
GREETINGS_PATTERN = re.compile(r"^\s*(hi|hello|hey|hii|hiya|good morning|good afternoon|good evening)\b", flags=re.I)

# Keywords mapping for intent to section
SECTION_KEYWORDS = {
    "side_effects": ["side effect", "side effects", "adverse reaction", "adverse reactions", "adverse"],
    "benefits": ["benefit", "benefits", "indication", "indications", "use"],
    "dosage": ["dose", "dosage", "how often", "how much"],
    "composition": ["composition", "ingredient", "contains"],
    "contraindications": ["contraindication", "contraindications"],
}


def is_greeting(text: str) -> bool:
    return bool(GREETINGS_PATTERN.match(text.strip()))


def simple_gibberish_check(text: str) -> bool:
    """
    Heuristic to detect gibberish: many non-alpha tokens or extremely short nonsense.
    Returns True if text looks like gibberish (should be rejected).
    """
    if not text or len(text.strip()) < 2:
        return True
    tokens = re.findall(r"[A-Za-z']+", text)
    if not tokens:
        return True
    # fraction of tokens with at least one vowel
    vowels = set("aeiou")
    vowel_tokens = sum(1 for t in tokens if any(ch in vowels for ch in t.lower()))
    frac = vowel_tokens / max(1, len(tokens))
    if frac < 0.4:
        return True
    # avoid single-character repeated strings
    if re.match(r"^[a-z]{1,3}$", text.strip(), flags=re.I):
        return False
    # otherwise not gibberish
    return False


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def detect_intent_section(query: str) -> str:
    q = query.lower()
    for section, kwlist in SECTION_KEYWORDS.items():
        for kw in kwlist:
            if kw in q:
                return section
    return ""


def call_ollama_completion(model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0):
    """
    Optional: Call Ollama local server. Requires Ollama running and the model present.
    Expects OLLAMA_HOST env var like 'http://localhost:11434' or defaults to http://localhost:11434
    """
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host}/api/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Ollama call failed:", e)
        return None


def format_sources(selected_chunks: List[Dict]) -> str:
    """
    Create short citation text from selected chunk metadata.
    """
    lines = []
    for c in selected_chunks:
        md = c["metadata"]
        src = md.get("source_file", md.get("product", "unknown"))
        page = md.get("page", "")
        section = md.get("section", "")
        lines.append(f"{src} | page:{page} | section:{section}")
    return "\n".join(lines)