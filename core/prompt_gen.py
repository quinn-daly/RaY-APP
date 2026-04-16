"""
LLM-based prompt generation for Metabolic Prompt Studio.

Generates 12 architecturally rich prompts (4 lenses × 3 specificity levels)
from a seminal intention using a live language model.

Provider:  OpenAI gpt-4o-mini  (~$0.001 per full run of 12 prompts)
Fallback:  core.prompt_sim.generate_prompts  (deterministic, no API required)

Environment:
    OPENAI_API_KEY — set this to enable live generation;
                     absent key falls back silently to the deterministic simulator
"""
from __future__ import annotations

import json
import os
from typing import List

from core.models import Prompt


# ---------------------------------------------------------------------------
# Specificity level guidance — injected into the system prompt
# ---------------------------------------------------------------------------

_SPEC_GUIDE = {
    "low": (
        "Abstract and conceptual. 40–70 words. "
        "No specific materials, dimensions, or programs. "
        "Spatial conditions, relational logics, conceptual tensions. "
        "Write as an architectural theorist."
    ),
    "medium": (
        "Spatial and material. 70–110 words. "
        "Specific spatial sequences, material qualities, light conditions, atmospheric register. "
        "Architecturally grounded but not dimensionally specified. "
        "Write as an architect presenting a design."
    ),
    "high": (
        "Technical and precise. 110–160 words. "
        "Structural systems, material assembly, tectonic logic, "
        "dimensional and performance specifics. "
        "Dense, professionally calibrated. "
        "Write as a technical architect or critic."
    ),
}


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(lenses: list) -> str:
    lens_lines = "\n".join(
        f"  {l['lens_id']}. {l['lens_name']}: {l.get('description', '')}"
        for l in lenses
    )
    spec_lines = "\n".join(
        f"  {spec}: {guide}" for spec, guide in _SPEC_GUIDE.items()
    )
    lens_names_json = json.dumps([l["lens_name"] for l in lenses])
    return (
        "You are an architectural research assistant generating image prompts for a design study tool.\n"
        "Each prompt simultaneously guides an AI image model toward a specific architectural vision "
        "AND functions as a piece of architectural analysis text.\n\n"
        f"LENSES — four analytical frames, each refracting the same idea differently:\n"
        f"{lens_lines}\n\n"
        f"SPECIFICITY LEVELS — three registers of resolution within each lens:\n"
        f"{spec_lines}\n\n"
        "RULES:\n"
        "- Every prompt must emerge genuinely from the seminal intention — no generic architectural clichés\n"
        "- Each prompt must be visually evocative and spatially specific enough to anchor an image\n"
        "- Within a lens, the three prompts should progress coherently: abstract → spatial → technical\n"
        "- Across lenses, prompts should feel intellectually distinct — the same idea refracted differently\n"
        "- Write in the third person, present tense, as spatial description\n"
        "- Do not begin with the phrase 'This space' or 'This building'\n\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON object with a single key 'prompts' containing exactly 12 objects.\n"
        "Generate in this order: all 3 specificities for lens 1, then lens 2, then lens 3, then lens 4.\n"
        f"Lens names must match exactly: {lens_names_json}\n"
        "Specificity values must be exactly: \"low\", \"medium\", or \"high\"\n\n"
        'Schema: {"prompts": [{"lens_name": "...", "specificity": "low|medium|high", "text": "..."}, ...]}'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_prompts(intention: str, lenses: list) -> List[Prompt]:
    """
    Generate 12 Prompt objects using OpenAI gpt-4o-mini.

    Falls back to the deterministic prompt_sim generator if:
      - OPENAI_API_KEY is not set
      - the openai package is not installed
      - the API call fails for any reason

    The returned list is structurally identical to prompt_sim output — same
    Prompt fields, same p01–p12 IDs — so all downstream code is unaffected.

    Args:
        intention: Seminal intention text entered by the user.
        lenses:    List of lens dicts with keys lens_id, lens_name, description.

    Returns:
        List of 12 Prompt objects.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        from core.prompt_sim import generate_prompts as _sim
        return _sim(intention, lenses)

    try:
        return _call_llm(intention, lenses, api_key)
    except Exception:
        from core.prompt_sim import generate_prompts as _sim
        return _sim(intention, lenses)


# ---------------------------------------------------------------------------
# Internal: live LLM call + response parsing
# ---------------------------------------------------------------------------

def _call_llm(intention: str, lenses: list, api_key: str) -> List[Prompt]:
    try:
        import openai as _openai
    except ImportError:
        raise RuntimeError("openai package not installed — run: pip install openai")

    client = _openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _build_system_prompt(lenses)},
            {
                "role": "user",
                "content": (
                    f'Seminal intention: "{intention}"\n\n'
                    "Generate the 12 architectural image prompts now."
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.85,   # enough creative range without chaos
        max_tokens=3000,
    )

    raw = response.choices[0].message.content
    return _parse_response(raw, lenses)


def _parse_response(raw: str, lenses: list) -> List[Prompt]:
    data  = json.loads(raw)
    items = data.get("prompts", [])

    if len(items) != 12:
        raise ValueError(f"Expected 12 prompts from LLM, received {len(items)}")

    lens_by_name = {l["lens_name"]: l for l in lenses}
    valid_specs  = {"low", "medium", "high"}

    prompts: List[Prompt] = []
    for i, item in enumerate(items, start=1):
        lname = item.get("lens_name", "").strip()
        spec  = item.get("specificity", "").strip()
        text  = item.get("text", "").strip()

        if lname not in lens_by_name:
            raise ValueError(
                f"Item {i}: lens_name '{lname}' not in configured lenses"
            )
        if spec not in valid_specs:
            raise ValueError(
                f"Item {i}: specificity '{spec}' must be low, medium, or high"
            )
        if not text:
            raise ValueError(f"Item {i}: empty text field")

        lens = lens_by_name[lname]
        prompts.append(Prompt(
            prompt_id=f"p{i:02d}",
            cycle=i,
            lens_id=lens["lens_id"],
            lens_name=lname,
            specificity=spec,
            text=text,
            word_count=len(text.split()),
        ))

    return prompts
