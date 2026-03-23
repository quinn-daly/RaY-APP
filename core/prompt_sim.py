"""
Deterministic prompt simulator for Metabolic Prompt Studio.

Given a seminal intention and a list of lenses, generates 12 prompts
(4 lenses × 3 specificity levels). The same input always produces the
same output — no randomness, no external API calls.

Generation strategy:
  - Extract a short "concept phrase" from the intention (non-stopword words)
  - For each lens × specificity, SHA-256-hash the key to get a seed
  - Use the seed bytes to select 3–5 sentences from a per-lens/specificity bank
  - Inject the concept phrase into {concept} placeholders
"""

from __future__ import annotations

import hashlib
import re
import string
from typing import List

from core.models import Prompt

# ---------------------------------------------------------------------------
# Default lenses
# ---------------------------------------------------------------------------

DEFAULT_LENSES = [
    {
        "lens_id": 1,
        "lens_name": "Parsed Complexity",
        "description": "Decompose the intention into its constituent components and structural relationships.",
    },
    {
        "lens_id": 2,
        "lens_name": "Surfaced Assumptions",
        "description": "Identify and interrogate the unstated premises that shape the design territory.",
    },
    {
        "lens_id": 3,
        "lens_name": "Multiple Perspectives",
        "description": "Examine the intention through the eyes of different stakeholders and disciplines.",
    },
    {
        "lens_id": 4,
        "lens_name": "Logical Scaffolding",
        "description": "Reconstruct the underlying argumentative structure and its decision dependencies.",
    },
]

# ---------------------------------------------------------------------------
# Sentence banks  (lens_id → specificity → list of sentences)
# ---------------------------------------------------------------------------

_BANKS: dict[int, dict[str, list[str]]] = {
    1: {  # Parsed Complexity
        "low": [
            "The seminal intention of {concept} invites a first-order decomposition into its constituent forces.",
            "At the broadest level, {concept} can be understood through the tension between enclosure and exposure.",
            "The primary structural logic of {concept} emerges from the relationship between its load-bearing elements and its spatial voids.",
            "A reductive reading of {concept} reveals three core operations: organization, mediation, and threshold.",
            "The fundamental complexity within {concept} lies not in its individual parts, but in how those parts negotiate their shared adjacencies.",
            "Parsing {concept} begins with recognizing which elements are structural and which are merely symptomatic.",
            "The coarse grain of {concept} contains a latent order that resists first-glance legibility.",
        ],
        "medium": [
            "Parsing {concept} reveals a second-order system of nested hierarchies, where each scale of resolution introduces new dependencies.",
            "The structural grammar of {concept} operates through recurring typological units that appear simultaneously at multiple scales.",
            "Between the macro-organization of {concept} and its micro-detail lies a middle-ground logic that frequently determines the character of the whole.",
            "Component analysis of {concept} shows that complexity is unevenly distributed — certain nodes carry disproportionate organizational weight.",
            "The interfaces between subsystems in {concept} are where latent complexity becomes visible and must be actively resolved.",
            "A mid-resolution reading of {concept} reveals the connective tissue: the transitions, thresholds, and gradient zones that bind its major elements.",
            "The organizational logic of {concept} shifts register at specific moments of scale transition, and those transitions are where the design's character is formed.",
        ],
        "high": [
            "A granular decomposition of {concept} exposes nested decision-trees: each element is both autonomous and load-bearing within a larger syntactic chain.",
            "The precise structural logic of {concept} can be mapped as a directed graph in which each node carries both a local function and a systemic consequence.",
            "High-resolution parsing of {concept} reveals micro-discontinuities — moments where the organizational logic shifts register or switches from additive to subtractive operation.",
            "At the level of material assembly, {concept} encodes structural decisions that are invisible at higher scales but determine the thermal, acoustic, and spatial performance of the whole.",
            "A full complexity map of {concept} distinguishes between essential complexity (inherent to the problem itself) and accidental complexity (introduced by the chosen solution strategy).",
            "The granular syntax of {concept} contains moments of controlled indeterminacy — zones where precise specification gives way to adaptive tolerance, and the design absorbs variation rather than resisting it.",
            "Micro-scale analysis of {concept} exposes the grain of its making: the sequences of assembly, the tolerances between components, and the material logics that quietly govern what is possible at larger scales.",
        ],
    },
    2: {  # Surfaced Assumptions
        "low": [
            "{concept} rests on a set of unstated assumptions about what this kind of work is for and who it ultimately serves.",
            "Before any design moves can be made within {concept}, the inherited assumptions of its typological context must first be named.",
            "The most powerful constraints shaping {concept} are the ones that have not yet been recognized as constraints at all.",
            "An assumption audit of {concept} begins with the question: what would have to be false for this approach to fail entirely?",
            "The default logic embedded in {concept} encodes cultural, economic, and disciplinary assumptions about what seems possible and what seems inevitable.",
            "Much of what appears as necessity within {concept} is actually convention — and conventions can be changed when they are first made visible.",
            "The unexamined ground of {concept} is where its most consequential decisions are quietly made.",
        ],
        "medium": [
            "Surfacing the assumptions within {concept} reveals a latent hierarchy of values: efficiency over experience, legibility over complexity, resolution over productive ambiguity.",
            "The design territory of {concept} is bounded by assumptions about scale, program, and user behavior that may not survive contact with the actual context of use.",
            "Three categories of assumptions structure {concept}: assumptions about the user, assumptions about the site, and assumptions about the brief itself.",
            "When the assumptions underlying {concept} are made explicit, new degrees of freedom become visible — territories previously hidden by conventional thinking.",
            "The productive tension in {concept} lies between its stated goals and the unstated assumptions that quietly constrain how those goals can be pursued.",
            "Distinguishing load-bearing assumptions from decorative ones within {concept} is the critical first move — only the former constrain the design space, and only the former deserve to be challenged.",
            "The middle register of assumptions within {concept} concerns social behavior: who is expected to do what, when, and with what degree of agency — expectations that the design will either reinforce or quietly subvert.",
        ],
        "high": [
            "A rigorous assumption audit of {concept} distinguishes between first-order assumptions (explicit in the brief), second-order assumptions (encoded in typological convention), and third-order assumptions (embedded in disciplinary training itself).",
            "The most consequential assumptions within {concept} concern the relationship between spatial organization and social behavior — an assumption that practice has historically treated as settled but which remains deeply contested.",
            "Surfacing the economic assumptions embedded in {concept} reveals that certain design options are precluded not by technical necessity but by financial convention masquerading as architectural constraint.",
            "A genealogical reading of the assumptions in {concept} traces how historical precedents have been silently normalized into present-day standards, foreclosing alternatives that remain technically and socially viable.",
            "The operational assumptions of {concept} — about maintenance cycles, user agency, and systemic adaptability — are often where the longest-term consequences of design decisions accumulate, invisible at the moment of conception.",
            "A full assumption map of {concept} would surface not just what is assumed to be true, but what is assumed to be unchangeable — the naturalized constraints that close off entire regions of the solution space before design even begins.",
            "The deepest layer of assumption in {concept} concerns temporality: what timescale the design is optimized for, whose future it imagines, and which kinds of change it is designed to absorb versus which it is designed to prevent.",
        ],
    },
    3: {  # Multiple Perspectives
        "low": [
            "{concept} appears differently depending on where you stand in relation to it — literally and figuratively.",
            "The same spatial logic of {concept} yields fundamentally different experiences for those who move through it versus those who maintain it over time.",
            "A multiperspectival reading of {concept} begins by identifying whose perspective has been centered — and whose has been treated as secondary — in its formulation.",
            "Different disciplines will frame {concept} in fundamentally different terms: what engineering calls a load path, experience design calls a journey, and sociology calls a corridor of power.",
            "The tensions within {concept} become visible only when multiple, potentially incompatible perspectives are held in relation simultaneously.",
            "Every design decision within {concept} is simultaneously a social decision — one that distributes advantage and disadvantage across different groups of people.",
            "The coarsest multiperspectival move within {concept} is simply to ask: for whom is this optimized, and at whose expense?",
        ],
        "medium": [
            "Reading {concept} through the lens of the first-time visitor, the daily inhabitant, and the maintenance worker reveals three distinct spatial logics operating in parallel.",
            "The institutional perspective on {concept} prioritizes legibility and control; the individual perspective prioritizes agency and orientation. The design task is to hold both without collapsing either.",
            "A cross-disciplinary reading of {concept} — architectural, ecological, social, and economic — reveals that optimization from any single perspective produces suboptimal outcomes for the whole.",
            "The political economy of {concept} looks very different from the perspective of those who commission it versus those who inhabit it versus those who are excluded from inhabiting it.",
            "Temporal perspective shifts the reading of {concept} fundamentally: what appears resolved at the moment of occupation looks different at five years, twenty years, and across a generation.",
            "Holding the designer's perspective alongside the user's perspective within {concept} is not a synthesis task — it is a translation task, and something is always lost and gained in translation.",
            "The ecological perspective on {concept} introduces non-human stakeholders whose timescales, spatial ranges, and vulnerability profiles are incommensurable with those of human occupants — a genuine design challenge, not a decorative concern.",
        ],
        "high": [
            "A structured multiperspectival analysis of {concept} requires mapping at minimum six distinct stakeholder positions: client, user, constructor, regulator, neighbor, and future inhabitant — each carrying incommensurable success criteria.",
            "The phenomenological perspective on {concept} attends to bodily experience, sensory sequence, and affective tone; the systems perspective attends to flows, cycles, and infrastructural interdependencies. A complete understanding requires both without collapsing either into the other.",
            "Reading {concept} through postcolonial, feminist, and disability-justice frameworks reveals that its default spatial assumptions carry normative implications that are often invisible to the dominant professional culture producing it.",
            "A game-theoretic perspective on {concept} models the incentive structures of each stakeholder and identifies where individually rational behavior produces collectively irrational outcomes — a key site for deliberate design intervention.",
            "The long-term ecological perspective on {concept} foregrounds material lifecycles, embodied carbon, and adaptive reuse potential, producing design priorities that frequently conflict with short-term aesthetic or economic imperatives and demand explicit negotiation.",
            "A deep multiperspectival analysis of {concept} must account for perspectives that are not yet representable — future users, non-existent communities, ecosystems in transition — requiring the design to encode a kind of prospective humility.",
            "The conflict between perspectival readings of {concept} is not a problem to be resolved but a generative condition to be maintained: the design that holds multiple valid interpretations simultaneously is more robust than the design optimized for a single reading.",
        ],
    },
    4: {  # Logical Scaffolding
        "low": [
            "The underlying logic of {concept} can be reconstructed as a sequence of nested if-then propositions, each dependent on the one preceding it.",
            "{concept} rests on a foundational axiom that determines which questions can be legitimately asked and which must remain unaddressed.",
            "Before the design of {concept} can proceed coherently, its logical prerequisites must be established and ordered.",
            "The argument embedded in {concept} has a structure: a premise, a condition, an operation, and a consequence — and each element can be independently interrogated.",
            "Logical scaffolding for {concept} means building a reasoning structure robust enough to support decisions under uncertainty without collapsing into contradiction.",
            "The first move in constructing the scaffold for {concept} is to separate description from prescription — what is observed from what is proposed.",
            "The logical spine of {concept} is rarely visible in the finished design, but it determines the coherence of every decision made along the way.",
        ],
        "medium": [
            "The logical architecture of {concept} can be described as a decision tree in which early choices constrain the solution space for all subsequent moves.",
            "Building the scaffolding for {concept} requires distinguishing between arguments that are logically necessary and those that are merely conventionally accepted as such.",
            "The inferential chain within {concept} moves from observation to typological precedent to design principle to spatial move — each step carrying assumptions that should be made explicit and tested.",
            "A logical map of {concept} identifies which of its propositions are mutually supporting and which are in productive tension, requiring negotiation rather than forced resolution.",
            "The generative logic of {concept} operates through a series of transformations applied to an initial schema — and the choice of that initial schema is itself a decision that demands explicit justification.",
            "Tracing the logical dependencies of {concept} reveals which decisions are truly foundational and which are downstream consequences that could be changed without disturbing the core structure.",
            "The scaffold of {concept} must account for the difference between arguments that are valid (follow logically from their premises) and arguments that are sound (whose premises are actually true) — a distinction that practice frequently collapses.",
        ],
        "high": [
            "Constructing the full logical scaffolding for {concept} requires operating simultaneously at three registers: the ontological (what kinds of things exist here), the epistemological (what can be known and how), and the normative (what should be done and why).",
            "The logical structure of {concept} can be formalized as a set of axioms, inference rules, and derived propositions — a formalization that makes the system's commitments visible and open to targeted, principled critique.",
            "High-resolution logical analysis of {concept} distinguishes between its deductive structure (what must follow given its premises) and its inductive warrant (what empirical evidence supports its central claims), revealing where the argument is strongest and where it relies on extrapolation.",
            "The scaffolding of {concept} must account for the difference between synchronic logic (the structure as it exists at a given moment) and diachronic logic (the structure as it unfolds and transforms across time), since these two registers often operate by mutually incompatible rules.",
            "A rigorous logical reconstruction of {concept} exposes the points where the argument shifts from descriptive to prescriptive register — transitions that are often unmarked in practice but that carry the full normative weight of the design proposal.",
            "The most fragile points in the logical scaffold of {concept} are typically the analogical moves — places where a structure from one domain is imported into another without sufficient justification for why the mapping holds across the difference.",
            "Full logical transparency for {concept} requires not just stating the conclusions but reconstructing the full inferential path: the observations that motivated the premises, the principles that licensed the inferences, and the scope conditions that limit where the conclusions apply.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Stopwords (used for concept extraction)
# ---------------------------------------------------------------------------

_STOP = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "that",
    "this", "these", "those", "it", "its", "as", "up", "into", "about",
    "through", "between", "how", "what", "which", "who", "when", "where",
    "not", "no", "i", "we", "they", "their", "our", "my", "your",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_concept(intention: str, max_words: int = 3) -> str:
    """
    Return a short concept phrase (≤3 words) from the seminal intention.
    Kept short so it reads naturally when injected into sentence templates.
    E.g. "The vertical inhabitation of collective memory…" → "vertical inhabitation memory"
    """
    words = re.sub(r"[^\w\s]", "", intention.lower()).split()
    key = [w for w in words if w not in _STOP and len(w) > 2]
    return " ".join(key[:max_words]) if key else intention[:40]


def generate_prompts(intention: str, lenses: list) -> List[Prompt]:
    """
    Generate 12 Prompt objects (4 lenses × 3 specificities).
    Deterministic: same inputs → same outputs.
    """
    concept = extract_concept(intention)
    specificities = ["low", "medium", "high"]
    prompts: List[Prompt] = []
    cycle = 1

    for lens in lenses:
        lid = lens["lens_id"]
        lname = lens["lens_name"]
        bank = _BANKS.get(lid, _BANKS[1])  # fallback to lens 1 if custom

        for spec in specificities:
            text = _build_prompt(intention, concept, lid, lname, spec, bank)
            prompts.append(
                Prompt(
                    prompt_id=f"p{cycle:02d}",
                    cycle=cycle,
                    lens_id=lid,
                    lens_name=lname,
                    specificity=spec,
                    text=text,
                    word_count=len(text.split()),
                )
            )
            cycle += 1

    return prompts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_prompt(
    intention: str,
    concept: str,
    lens_id: int,
    lens_name: str,
    specificity: str,
    bank: dict,
) -> str:
    """Select and assemble sentences from the bank deterministically."""
    sentences = bank.get(specificity, bank["low"])
    n = len(sentences)

    # Seed bytes from hash of intention + lens + specificity
    key = f"{intention}|{lens_name}|{specificity}"
    seed = hashlib.sha256(key.encode("utf-8")).digest()

    # Select 4 sentence indices deterministically
    indices = []
    for i in range(4):
        idx = seed[i % len(seed)] % n
        # Avoid duplicate sentences
        attempts = 0
        while idx in indices and attempts < n:
            idx = (idx + 1) % n
            attempts += 1
        indices.append(idx)

    selected = [sentences[i].replace("{concept}", concept) for i in indices]
    return " ".join(selected)
