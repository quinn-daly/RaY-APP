"""
Vocabulary analysis for Metabolic Prompt Studio.

Tokenizes prompt texts (and optionally intervention notes),
removes stopwords, and computes frequency distributions
overall, by lens, and by specificity.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional

from core.models import ImageRecord, Prompt

# ---------------------------------------------------------------------------
# Stopword list (English, architecture-domain additions)
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "get", "gets", "got", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "will", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves",
    # short/common words not caught above
    "one", "two", "three", "four", "five", "first", "second", "third",
    "make", "made", "take", "taken", "give", "given", "just", "may", "might",
    "must", "shall", "well", "still", "often", "rather", "whether", "without",
    "within", "between", "across", "however", "therefore", "thus", "hence",
    "yet", "even", "already", "never", "always", "every", "each",
    "either", "neither", "both", "such", "many", "much", "any", "each",
    "where", "here", "there", "now", "then", "when", "how", "its",
}

_PUNC_RE = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split, remove stopwords and short tokens."""
    cleaned = _PUNC_RE.sub(" ", text.lower())
    return [
        w for w in cleaned.split()
        if w not in STOPWORDS and len(w) > 2
    ]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_vocab(
    prompts: List[Prompt],
    image_records: Optional[List[ImageRecord]] = None,
    include_interventions: bool = False,
) -> Dict:
    """
    Returns a dict with:
        overall_freq:   Counter  (word → count across all included prompts)
        by_lens:        dict[lens_name → Counter]
        by_specificity: dict[specificity → Counter]
        total_tokens:   int  (total non-stopword tokens, for % calculation)
        rows:           list[dict]  — flat rows for CSV export, one row per
                        (word, group) combination so the file is analysis-ready
    """
    included = [p for p in prompts if not p.excluded]

    overall: Counter = Counter()
    by_lens: Dict[str, Counter] = {}
    by_spec: Dict[str, Counter] = {}

    for p in included:
        tokens = tokenize(p.text)
        overall.update(tokens)

        by_lens.setdefault(p.lens_name, Counter()).update(tokens)
        by_spec.setdefault(p.specificity, Counter()).update(tokens)

    if include_interventions and image_records:
        for rec in image_records:
            if rec.intervention_note:
                overall.update(tokenize(rec.intervention_note))

    total = sum(overall.values()) or 1   # avoid division by zero

    # Build flat rows: overall + per-lens + per-specificity
    # Each row: word | count | pct | group | group_value
    rows: List[dict] = []

    for word, count in overall.most_common():
        rows.append({
            "word": word,
            "count": count,
            "pct": round(count / total * 100, 2),
            "group": "overall",
            "group_value": "all",
        })

    for lens_name, counter in by_lens.items():
        lens_total = sum(counter.values()) or 1
        for word, count in counter.most_common(30):
            rows.append({
                "word": word,
                "count": count,
                "pct": round(count / lens_total * 100, 2),
                "group": "lens",
                "group_value": lens_name,
            })

    for spec, counter in by_spec.items():
        spec_total = sum(counter.values()) or 1
        for word, count in counter.most_common(30):
            rows.append({
                "word": word,
                "count": count,
                "pct": round(count / spec_total * 100, 2),
                "group": "specificity",
                "group_value": spec,
            })

    return {
        "overall_freq": overall,
        "by_lens": by_lens,
        "by_specificity": by_spec,
        "total_tokens": total,
        "rows": rows,
    }


def top_n(counter: Counter, n: int = 20) -> List[tuple]:
    """Return top-n (word, count) pairs."""
    return counter.most_common(n)


# ---------------------------------------------------------------------------
# Drift analysis (recurrence)
# ---------------------------------------------------------------------------

def analyze_drift(
    original_prompts: List[Prompt],
    recurrent_records: List["ImageRecord"],
    seminal_intention: str,
) -> Dict:
    """
    Compare vocabulary between the original refracted prompts and the
    evolving recurrent mutations to surface how language is transforming.

    Returns:
        anchor_terms    — tokens from the seminal intention still present in recurrent prompts
        emerging_terms  — new tokens in recurrent prompts absent from original prompts
        fading_terms    — tokens prominent in originals but diminishing in recurrents
        original_freq   — Counter of original prompt tokens
        recurrent_freq  — Counter of recurrent prompt tokens (from current_prompt_text)
        drift_over_time — list of {iteration, avg_similarity} for charting
    """
    intention_tokens = set(tokenize(seminal_intention))

    # Original vocabulary — from the 12 Phase-1 prompts
    original_freq: Counter = Counter()
    for p in [p for p in original_prompts if not p.excluded]:
        original_freq.update(tokenize(p.text))

    # Recurrent vocabulary — from current_prompt_text of each recurrent ImageRecord
    # Deduplicate by text to avoid counting the same mutation state multiple times
    recurrent_freq: Counter = Counter()
    seen_texts: set[str] = set()
    for rec in recurrent_records:
        text = rec.current_prompt_text or rec.parent_prompt_text
        if text and text not in seen_texts:
            recurrent_freq.update(tokenize(text))
            seen_texts.add(text)

    original_set  = set(original_freq.keys())
    recurrent_set = set(recurrent_freq.keys())

    # Anchor: from the seminal intention and still present in recurrent prompts
    anchor_terms = sorted(intention_tokens & recurrent_set)

    # Emerging: appear in recurrent prompts but were absent from original prompts
    emerging_raw = recurrent_set - original_set
    emerging_terms = Counter({w: recurrent_freq[w] for w in emerging_raw}).most_common(20)

    # Fading: present in originals but significantly reduced in recurrents
    # Threshold: recurrent count < 30% of original count
    fading_terms = sorted(
        w for w in original_set
        if original_freq[w] > 0
        and recurrent_freq.get(w, 0) < original_freq[w] * 0.30
    )[:20]

    # Drift over time: group ImageRecords by generation_iteration
    # Each group gives the average semantic_similarity for that iteration round
    by_iteration: Dict[int, list[float]] = {}
    for rec in recurrent_records:
        by_iteration.setdefault(rec.generation_iteration, []).append(rec.semantic_similarity)

    drift_over_time = [
        {"iteration": it, "avg_similarity": round(sum(sims) / len(sims), 3)}
        for it, sims in sorted(by_iteration.items())
    ]

    return {
        "anchor_terms":    anchor_terms,
        "emerging_terms":  emerging_terms,
        "fading_terms":    fading_terms,
        "original_freq":   original_freq,
        "recurrent_freq":  recurrent_freq,
        "drift_over_time": drift_over_time,
    }
