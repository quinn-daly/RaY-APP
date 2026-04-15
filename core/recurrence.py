"""
Recurrence Engine — Metabolic Prompt Studio.

Drives continuous transformation of prompt strands, extending each run from a
fixed 3-phase experiment into an ongoing metabolic process.

Each recurrence cycle:
  1. Receives the current evolving text for one prompt strand
  2. Applies a controlled mutation drawn from the same curated sentence banks
     that generated the original prompt — so all language stays architecturally grounded
  3. Generates a new image event via image_sim (placeholder) or live API (future)
  4. Returns a RecurrenceStep with full lineage data attached to the ImageRecord

Mutation types:
  transpose  — same specificity level, different sentence selection (minimal drift)
  intensify  — shift toward higher specificity (increases technical and conceptual density)
  contract   — shift toward lower specificity (abstracts back toward essence)
  reframe    — mirror to opposite end of specificity scale (semantic rupture)
  expand     — append one new sentence from the current bank (grows complexity)

Mutation intensity governs which types are available per iteration:
  low    → [transpose]
  medium → [transpose, intensify, contract]
  high   → [intensify, reframe, expand, contract, transpose]

The sentence banks in prompt_sim are the shared vocabulary source. By drawing all
mutations from these banks, every evolved prompt remains traceable to its lens and
to the architectural discourse that the banks encode. The system does not drift into
noise — it drifts within a bounded conceptual territory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from core.models import ImageRecord, Prompt
from core.vocab import tokenize
from core.prompt_sim import _BANKS, extract_concept
from core.image_providers import ImageProvider, get_provider


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RecurrenceStep:
    """All outputs from a single recurrence cycle."""
    mutated_text:          str
    mutation_type:         str
    mutation_note:         str
    similarity_to_origin:  float    # Jaccard vs the original Phase-1 prompt text
    image_record:          ImageRecord


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Jaccard similarity between the token sets of two texts.
    Returns 1.0 for identical vocabulary, 0.0 for fully disjoint.
    Suitable as a lightweight drift score without requiring embeddings.
    """
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return round(len(tokens_a & tokens_b) / len(tokens_a | tokens_b), 3)


# ---------------------------------------------------------------------------
# Mutation strategy selection
# ---------------------------------------------------------------------------

_INTENSITY_TYPES: dict[str, list[str]] = {
    "low":    ["transpose"],
    "medium": ["transpose", "intensify", "contract"],
    "high":   ["intensify", "reframe", "expand", "contract", "transpose"],
}


def _select_mutation_type(iteration: int, intensity: str) -> str:
    """Cycle through mutation types deterministically based on iteration and intensity."""
    options = _INTENSITY_TYPES.get(intensity, _INTENSITY_TYPES["medium"])
    return options[iteration % len(options)]


# ---------------------------------------------------------------------------
# Mutation implementation
# ---------------------------------------------------------------------------

_SPEC_ORDER = ["low", "medium", "high"]


def _mutate_text(
    current_text: str,
    prompt: Prompt,
    mutation_type: str,
    iteration: int,
    concept: str,
) -> Tuple[str, str]:
    """
    Apply one controlled mutation to the current prompt text.
    Returns (new_text, mutation_note).

    All sentence material is drawn from the same banks used to generate the
    original prompt, so vocabulary stays within the lens's established register.
    The iteration value seeds a SHA-256 hash so each cycle advances to a
    different selection — mutations progress forward without repeating.
    """
    bank = _BANKS.get(prompt.lens_id, _BANKS[1])

    # Determine target specificity level for this mutation
    current_idx = _SPEC_ORDER.index(prompt.specificity) if prompt.specificity in _SPEC_ORDER else 1
    target_spec = {
        "transpose": prompt.specificity,
        "intensify": _SPEC_ORDER[min(current_idx + 1, 2)],
        "contract":  _SPEC_ORDER[max(current_idx - 1, 0)],
        "reframe":   _SPEC_ORDER[2 - current_idx],   # mirror to opposite end
        "expand":    prompt.specificity,
    }.get(mutation_type, prompt.specificity)

    sentences = bank[target_spec]
    n = len(sentences)

    # Seed varies by iteration so successive mutations advance to fresh selections
    key = f"{prompt.prompt_id}|{mutation_type}|{iteration}"
    seed = hashlib.sha256(key.encode()).digest()

    # ── contract: distil to two essential sentences ──────────────────────────
    if mutation_type == "contract":
        idx_a = seed[0] % n
        idx_b = (seed[0] + seed[1]) % n
        if idx_b == idx_a:
            idx_b = (idx_b + 1) % n
        selected = [sentences[i].replace("{concept}", concept) for i in (idx_a, idx_b)]
        note = f"Contracted: abstracted to {target_spec}-specificity essence ({len(selected)} sentences)"
        return " ".join(selected), note

    # ── expand: preserve current text, append one new sentence ───────────────
    if mutation_type == "expand":
        new_idx = seed[0] % n
        # Walk until we find a sentence not already approximated in the current text
        for _ in range(n):
            candidate = sentences[new_idx].replace("{concept}", concept)
            if candidate[:40] not in current_text:
                break
            new_idx = (new_idx + 1) % n
        candidate = sentences[new_idx].replace("{concept}", concept)
        # Cap growth at 6 sentences; rotate oldest out once cap is reached
        existing = [s.strip() for s in current_text.rstrip(".").split(". ") if s.strip()]
        if len(existing) < 6:
            new_text = current_text.rstrip(". ") + ". " + candidate
        else:
            new_text = ". ".join(existing[1:]) + ". " + candidate
        note = f"Expanded: appended new {target_spec}-specificity clause"
        return new_text, note

    # ── transpose / intensify / reframe: full 4-sentence rebuild ─────────────
    idxs: list[int] = []
    for i in range(4):
        idx = seed[i % 32] % n
        attempts = 0
        while idx in idxs and attempts < n:
            idx = (idx + 1) % n
            attempts += 1
        idxs.append(idx)
    selected = [sentences[i].replace("{concept}", concept) for i in idxs]
    note = {
        "transpose": f"Transposed: new {target_spec}-specificity selection within same register",
        "intensify": f"Intensified: shifted to {target_spec}-specificity register",
        "reframe":   f"Reframed: inverted to {target_spec}-specificity counter-orientation",
    }.get(mutation_type, f"Mutated via {mutation_type}")

    return " ".join(selected), note


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_recurrence_step(
    prompt: Prompt,
    original_prompt_text: str,
    seminal_intention: str,
    iteration: int,
    intensity: str,
    current_text: str,
    run_id: str,
    provider: Optional[ImageProvider] = None,
) -> RecurrenceStep:
    """
    Execute one recurrence cycle for a prompt strand.

    Args:
        prompt:               The Prompt dataclass (carries lens_id, specificity, prompt_id)
        original_prompt_text: Immutable — the text as generated in Phase 1 (never changes)
        seminal_intention:    The run's seminal intention (used for concept phrase extraction)
        iteration:            Global recurrence counter; drives mutation variety
        intensity:            "low" | "medium" | "high" — governs mutation type pool
        current_text:         The evolving text for this strand before this step
        run_id:               Run identifier for image file paths
        provider:             ImageProvider to use; defaults to the registered "sim" provider.

    Returns:
        RecurrenceStep containing the mutated text, mutation metadata, and a
        fully populated ImageRecord with lineage fields set.
    """
    if provider is None:
        provider = get_provider("sim")

    concept        = extract_concept(seminal_intention)
    mutation_type  = _select_mutation_type(iteration, intensity)
    mutated_text, mutation_note = _mutate_text(
        current_text, prompt, mutation_type, iteration, concept
    )

    similarity = compute_similarity(mutated_text, original_prompt_text)

    # Build image file path — "r" prefix distinguishes recurrent from original variations
    img_id   = f"{prompt.prompt_id}_r{iteration:04d}"
    rel_path = f"runs/{run_id}/images/{img_id}.png"

    img_result = provider.generate(
        prompt=mutated_text,
        output_path=Path(rel_path),
        variation_index=iteration % 4,
        lens_id=prompt.lens_id,
        lens_name=prompt.lens_name,
        specificity=prompt.specificity,
        intervention_note="",
    )

    record = ImageRecord(
        image_id=img_id,
        source_prompt_id=prompt.prompt_id,
        variation_index=iteration % 4,
        intervention_note="",
        image_path=img_result.image_path,
        pinned=False,
        created_at=datetime.now().isoformat(timespec="seconds"),
        user_note=None,
        # Lineage
        parent_prompt_text=current_text,
        current_prompt_text=img_result.prompt_used,
        mutation_note=mutation_note,
        semantic_similarity=similarity,
        generation_iteration=iteration,
        is_recurrent=True,
        provider_name=img_result.provider_name,
        raw_prompt_text=img_result.raw_prompt or mutated_text,
        generation_time_ms=img_result.generation_time_ms,
    )

    return RecurrenceStep(
        mutated_text=mutated_text,
        mutation_type=mutation_type,
        mutation_note=mutation_note,
        similarity_to_origin=similarity,
        image_record=record,
    )
