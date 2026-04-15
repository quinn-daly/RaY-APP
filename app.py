"""
Metabolic Prompt Studio — Streamlit App
A concept-demo workflow for architectural AI prompt design.

All outputs are SIMULATED. No external APIs are called.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from core.models import ImageRecord, Prompt, RunConfig
from core.prompt_sim import DEFAULT_LENSES, generate_prompts
from core.image_sim import generate_placeholder_image
from core.vocab import analyze_vocab, analyze_drift, top_n
from core import storage
from core.recurrence import run_recurrence_step, compute_similarity
from core.image_providers import (
    available_providers, get_provider, ProviderError,
    ROLLOUT_PROVIDERS, provider_label, provider_tier,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Metabolic Prompt Studio",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "run_id": None,
        "run_config": None,          # RunConfig
        "prompts": [],               # List[Prompt]
        "image_records": [],         # List[ImageRecord]
        "current_phase": 1,
        "prob_intervention": False,
        "include_interventions_vocab": False,
        "prompt_edits": {},
        # phase-2 intervention notes: {prompt_id: note_str}
        "intervention_notes": {},
        # image user notes: {image_id: note_str}
        "image_user_notes": {},
        # phase-2 navigator: index into _prompts_included()
        "p2_focused_idx": 0,
        # autosave feedback
        "_last_saved": None,
        # ── recurrence mode ──────────────────────────────────────────────────
        "recurrence_running":      False,
        "recurrence_paused":       False,
        "recurrence_intensity":    "medium",
        "recurrence_iteration":    0,
        "recurrence_strand_idx":   0,
        # per-strand evolving text: {prompt_id → current text}
        "recurrent_prompt_states": {},
        # per-strand immutable origin text: {prompt_id → original Phase-1 text}
        "original_prompt_texts":   {},
        # ── recurrence pacing + provider ────────────────────────────────────
        "recurrence_provider":         "sim",   # provider name string
        "recurrence_interval":         2,       # seconds between auto steps
        "recurrence_display_cadence":  1,       # re-render every N steps
        "recurrence_logging_cadence":  5,       # save to disk every N steps
        "recurrence_gen_since_save":   0,       # steps since last save
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_prompt_widget_cache(prompts) -> None:
    """Remove cached Streamlit widget values for prompt text areas.

    Must be called before st.rerun() after prompts are regenerated or a new
    run is loaded, otherwise the old textarea content stays visible.
    """
    for p in prompts:
        for key in (f"text_{p.prompt_id}", f"p2_text_{p.prompt_id}"):
            st.session_state.pop(key, None)


def _save_prompts() -> None:
    if st.session_state.run_id and st.session_state.prompts:
        storage.save_prompts(st.session_state.run_id, st.session_state.prompts)
        st.session_state._last_saved = datetime.now().strftime("%H:%M:%S")


def _save_records() -> None:
    if st.session_state.run_id:
        storage.save_image_records(st.session_state.run_id, st.session_state.image_records)
        st.session_state._last_saved = datetime.now().strftime("%H:%M:%S")


def _images_for_prompt(prompt_id: str) -> List[ImageRecord]:
    return [r for r in st.session_state.image_records if r.source_prompt_id == prompt_id]


def _total_images_generated() -> int:
    return len(st.session_state.image_records)


def _prompts_included() -> List[Prompt]:
    return [p for p in st.session_state.prompts if not p.excluded]


def _image_id(prompt_id: str, variation: int) -> str:
    return f"{prompt_id}_v{variation + 1}"


def _load_run_into_state(run_id: str) -> None:
    cfg, prompts, records = storage.load_run(run_id)
    st.session_state.run_id = run_id
    st.session_state.run_config = cfg
    st.session_state.prompts = prompts
    st.session_state.image_records = records
    # Restore intervention notes and user notes from records
    st.session_state.intervention_notes = {
        r.source_prompt_id: r.intervention_note
        for r in records if r.intervention_note
    }
    st.session_state.image_user_notes = {
        r.image_id: r.user_note
        for r in records if r.user_note
    }
    _clear_prompt_widget_cache(prompts)


def _do_generate_images(prompt: Prompt, variation_index: int, note: str) -> ImageRecord:
    """Generate one image and return a new ImageRecord (not yet appended)."""
    run_id = st.session_state.run_id
    img_id = _image_id(prompt.prompt_id, variation_index)
    fname = f"{img_id}.png"
    rel_path = f"runs/{run_id}/images/{fname}"
    abs_path = Path(rel_path)

    generate_placeholder_image(
        prompt_text=prompt.text,
        variation_index=variation_index,
        intervention_note=note,
        lens_id=prompt.lens_id,
        lens_name=prompt.lens_name,
        specificity=prompt.specificity,
        output_path=abs_path,
    )

    return ImageRecord(
        image_id=img_id,
        source_prompt_id=prompt.prompt_id,
        variation_index=variation_index,
        intervention_note=note,
        image_path=rel_path,
        pinned=False,
        created_at=datetime.now().isoformat(timespec="seconds"),
        user_note=None,
    )


# ---------------------------------------------------------------------------
# Recurrence engine — step execution and UI
# ---------------------------------------------------------------------------

def _run_one_recurrence_step() -> None:
    """
    Execute one recurrence cycle:
      1. Select the next prompt strand (round-robin through included prompts)
      2. Retrieve or initialise its current evolving text
      3. Delegate to core/recurrence.py for mutation + image generation
      4. Persist the new ImageRecord and advance counters
    """
    included = _prompts_included()
    if not included:
        return

    idx    = st.session_state.recurrence_strand_idx % len(included)
    prompt = included[idx]

    # Snapshot the original text the first time this strand enters recurrence
    if prompt.prompt_id not in st.session_state.original_prompt_texts:
        st.session_state.original_prompt_texts[prompt.prompt_id] = prompt.text

    original_text = st.session_state.original_prompt_texts[prompt.prompt_id]
    current_text  = st.session_state.recurrent_prompt_states.get(
        prompt.prompt_id, prompt.text
    )

    # Resolve provider; fall back to sim on any ProviderError
    provider_name = st.session_state.get("recurrence_provider", "sim")
    try:
        provider = get_provider(provider_name)
    except ProviderError:
        provider = get_provider("sim")
        st.session_state.recurrence_provider = "sim"

    try:
        step = run_recurrence_step(
            prompt=prompt,
            original_prompt_text=original_text,
            seminal_intention=st.session_state.run_config.seminal_intention,
            iteration=st.session_state.recurrence_iteration,
            intensity=st.session_state.recurrence_intensity,
            current_text=current_text,
            run_id=st.session_state.run_id,
            provider=provider,
        )
    except ProviderError:
        # Live provider failed mid-run; degrade silently to sim for this step
        fallback = get_provider("sim")
        step = run_recurrence_step(
            prompt=prompt,
            original_prompt_text=original_text,
            seminal_intention=st.session_state.run_config.seminal_intention,
            iteration=st.session_state.recurrence_iteration,
            intensity=st.session_state.recurrence_intensity,
            current_text=current_text,
            run_id=st.session_state.run_id,
            provider=fallback,
        )

    # Advance the strand's evolving text
    st.session_state.recurrent_prompt_states[prompt.prompt_id] = step.mutated_text

    # Persist according to logging cadence
    st.session_state.image_records.append(step.image_record)
    st.session_state.recurrence_gen_since_save = (
        st.session_state.recurrence_gen_since_save + 1
    )
    logging_cadence = st.session_state.get("recurrence_logging_cadence", 5)
    if st.session_state.recurrence_gen_since_save >= logging_cadence:
        _save_records()
        st.session_state.recurrence_gen_since_save = 0

    # Advance global counters
    st.session_state.recurrence_iteration  += 1
    st.session_state.recurrence_strand_idx  = (idx + 1) % len(included)


def _render_recurrence_section() -> None:
    """
    Recurrence mode UI — appended to the bottom of Phase 2.

    When not running: exposes a Start button and intensity selector.
    When running:     shows live controls, evolving strand states,
                      a recent-image feed, and auto-advances on each rerun.

    The auto-advance pattern (generate one step → st.rerun) uses Streamlit's
    own rerun mechanism as the loop clock. The user observes the system evolving
    and intervenes via Pause / Re-anchor / Stop when necessary.
    """
    st.divider()
    st.subheader("⟳ Recurrence Mode")

    if not st.session_state.recurrence_running:
        st.caption(
            "Transform the prompt strands continuously. "
            "The seminal intention persists; its refractions evolve."
        )

        col_btn, col_intensity = st.columns([2, 2])
        with col_btn:
            if st.button("Start Recurrence", type="primary", key="rec_start"):
                st.session_state.recurrence_running = True
                st.session_state.recurrence_paused  = False
                st.rerun()
        with col_intensity:
            intensity = st.select_slider(
                "Mutation intensity",
                options=["low", "medium", "high"],
                value=st.session_state.recurrence_intensity,
                key="rec_intensity_idle",
            )
            st.session_state.recurrence_intensity = intensity

        # Provider selector — rollout order
        st.markdown("**Image provider**")
        st.caption(
            "Tier 1 (sim) is free and instant. "
            "Move to Tier 2 (replicate_flux) for real imagery in research runs. "
            "Tier 3 (gemini_flash_image) adds a second live provider for comparison. "
            "Tier 4 (openai_gpt_image) is reserved for curated final runs."
        )
        current_p = st.session_state.get("recurrence_provider", "sim")
        if current_p not in ROLLOUT_PROVIDERS:
            current_p = "sim"
        tier_labels = [provider_label(p) for p in ROLLOUT_PROVIDERS]
        chosen_label = st.radio(
            "Select provider",
            options=tier_labels,
            index=ROLLOUT_PROVIDERS.index(current_p),
            key="rec_provider_idle",
            label_visibility="collapsed",
        )
        st.session_state.recurrence_provider = ROLLOUT_PROVIDERS[tier_labels.index(chosen_label)]

        # Pacing controls
        with st.expander("Pacing controls", expanded=False):
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                interval = st.number_input(
                    "Generation interval (s)",
                    min_value=0,
                    max_value=60,
                    value=st.session_state.get("recurrence_interval", 2),
                    step=1,
                    help="Seconds to wait between auto-advance steps.",
                    key="rec_interval_idle",
                )
                st.session_state.recurrence_interval = interval
            with p_col2:
                display_cadence = st.number_input(
                    "Display cadence (steps)",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get("recurrence_display_cadence", 1),
                    step=1,
                    help="Re-render the UI every N steps.",
                    key="rec_display_cadence_idle",
                )
                st.session_state.recurrence_display_cadence = display_cadence
            with p_col3:
                logging_cadence = st.number_input(
                    "Save cadence (steps)",
                    min_value=1,
                    max_value=50,
                    value=st.session_state.get("recurrence_logging_cadence", 5),
                    step=1,
                    help="Write image_log.json to disk every N steps.",
                    key="rec_logging_cadence_idle",
                )
                st.session_state.recurrence_logging_cadence = logging_cadence
        return

    # ── Running controls ──────────────────────────────────────────────────────
    col_pause, col_stop, col_anchor, col_intensity, col_provider_run = st.columns([1, 1, 2, 2, 3])

    with col_pause:
        if st.session_state.recurrence_paused:
            if st.button("▶ Resume", key="rec_resume"):
                st.session_state.recurrence_paused = False
                st.rerun()
        else:
            if st.button("⏸ Pause", key="rec_pause"):
                st.session_state.recurrence_paused = True
                st.rerun()

    with col_stop:
        if st.button("⏹ Finalize", key="rec_stop"):
            st.session_state.recurrence_running = False
            st.session_state.recurrence_paused  = False
            st.rerun()

    with col_anchor:
        if st.button("↺ Re-anchor to Original", key="rec_reanchor",
                     help="Reset all evolving strands back to their Phase-1 text"):
            st.session_state.recurrent_prompt_states = {}
            st.rerun()

    with col_intensity:
        intensity = st.select_slider(
            "Mutation intensity",
            options=["low", "medium", "high"],
            value=st.session_state.recurrence_intensity,
            key="rec_intensity_running",
        )
        st.session_state.recurrence_intensity = intensity

    with col_provider_run:
        current_p_run = st.session_state.get("recurrence_provider", "sim")
        if current_p_run not in ROLLOUT_PROVIDERS:
            current_p_run = "sim"
        tier_labels_run = [provider_label(p) for p in ROLLOUT_PROVIDERS]
        chosen_run = st.selectbox(
            "Switch provider",
            options=tier_labels_run,
            index=ROLLOUT_PROVIDERS.index(current_p_run),
            key="rec_provider_running",
            help="Takes effect on the next generation step.",
        )
        st.session_state.recurrence_provider = ROLLOUT_PROVIDERS[tier_labels_run.index(chosen_run)]

    # ── Status line ───────────────────────────────────────────────────────────
    recurrent_count = sum(1 for r in st.session_state.image_records if r.is_recurrent)
    status = "⏸ Paused" if st.session_state.recurrence_paused else "● Running"
    active_provider = st.session_state.get("recurrence_provider", "sim")
    st.caption(
        f"{status}  ·  Iteration {st.session_state.recurrence_iteration}"
        f"  ·  {recurrent_count} recurrent image{'s' if recurrent_count != 1 else ''} generated"
        f"  ·  provider: `{active_provider}`"
    )

    # ── Active strand states ──────────────────────────────────────────────────
    included = _prompts_included()
    evolved = {
        pid: txt for pid, txt in st.session_state.recurrent_prompt_states.items()
        if txt != st.session_state.original_prompt_texts.get(pid, "")
    }
    if evolved:
        with st.expander(f"Active strand states ({len(evolved)} evolved)", expanded=False):
            for p in included:
                current = st.session_state.recurrent_prompt_states.get(p.prompt_id)
                if not current or current == p.text:
                    continue
                original = st.session_state.original_prompt_texts.get(p.prompt_id, p.text)
                sim = compute_similarity(current, original)
                st.caption(
                    f"**{p.lens_name} · {p.specificity.capitalize()}**"
                    f"  —  similarity to origin: `{sim:.2f}`"
                )
                st.text(current[:220] + "…" if len(current) > 220 else current)
                st.divider()

    # ── Recent recurrent image feed ───────────────────────────────────────────
    recurrent_records = sorted(
        [r for r in st.session_state.image_records if r.is_recurrent],
        key=lambda r: r.created_at,
        reverse=True,
    )
    if recurrent_records:
        st.caption("Most recent recurrent images")
        cols = st.columns(4)
        for i, rec in enumerate(recurrent_records[:4]):
            with cols[i]:
                img_path = Path(rec.image_path)
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.warning(f"Missing: {img_path.name}")
                st.caption(f"`{rec.source_prompt_id}` · iter {rec.generation_iteration}")
                st.caption(f"*{rec.mutation_note}*")
                st.caption(
                    f"similarity: `{rec.semantic_similarity:.2f}`"
                    + (f"  ·  `{rec.provider_name}`" if rec.provider_name else "")
                )

    # ── Auto-advance: generate one step then rerun ────────────────────────────
    # Streamlit's rerun is the loop clock. Each execution renders state, generates
    # one step (or a display_cadence batch), sleeps for the configured interval,
    # then triggers the next execution. Pause / Finalize break the cycle.
    if not st.session_state.recurrence_paused and _prompts_included():
        display_cadence = max(1, st.session_state.get("recurrence_display_cadence", 1))
        interval        = st.session_state.get("recurrence_interval", 2)
        for _ in range(display_cadence):
            _run_one_recurrence_step()
            if st.session_state.recurrence_paused:
                break
        if interval > 0:
            time.sleep(interval)
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.title("⬡ Metabolic Prompt Studio")
    st.sidebar.caption("Demo mode — all outputs are simulated, no API calls are made.")
    st.sidebar.divider()

    # --- New Run ---
    with st.sidebar.expander("New Run", expanded=st.session_state.run_id is None):
        intention = st.text_area(
            "Seminal intention",
            placeholder="Enter your architectural design thesis…",
            height=100,
            key="sb_intention",
        )
        if st.button("Create Run", type="primary", disabled=not intention.strip()):
            cfg = storage.create_run(
                seminal_intention=intention.strip(),
                lenses=DEFAULT_LENSES,
            )
            st.session_state.run_id = cfg.run_id
            st.session_state.run_config = cfg
            st.session_state.prompts = []
            st.session_state.image_records = []
            st.session_state.prompt_edits = {}
            st.session_state.intervention_notes = {}
            st.session_state.image_user_notes = {}
            st.session_state.current_phase = 1
            st.session_state.p2_focused_idx = 0
            st.session_state._last_saved = None
            st.rerun()

    # --- Load Run ---
    runs = storage.list_runs()
    if runs:
        st.sidebar.divider()
        options = ["— select a run —"] + runs
        current_idx = (runs.index(st.session_state.run_id) + 1
                       if st.session_state.run_id in runs else 0)
        selected = st.sidebar.selectbox(
            "Load existing run",
            options=options,
            index=current_idx,
            key="sb_load_run",
        )
        if selected != "— select a run —" and selected != st.session_state.run_id:
            _load_run_into_state(selected)
            st.session_state.current_phase = 1
            st.session_state.p2_focused_idx = 0
            st.rerun()

    st.sidebar.divider()

    # --- Phase Navigation ---
    if st.session_state.run_id:
        st.sidebar.subheader(f"`{st.session_state.run_id}`")
        n_generated = _total_images_generated()
        n_pinned = sum(1 for r in st.session_state.image_records if r.pinned)
        n_prompts = len(st.session_state.prompts)
        st.sidebar.caption(f"Prompts: {n_prompts} · Images: {n_generated} · Pinned: {n_pinned}")

        # Build phase labels with completion indicators
        included = [p for p in st.session_state.prompts if not p.excluded]
        p1_done = bool(st.session_state.prompts)
        p2_done = bool(included) and n_generated >= len(included) * 4

        def _phase_label(x: int) -> str:
            labels = {
                1: ("1 — Prompt Refraction", p1_done),
                2: ("2 — Image Generation", p2_done),
                3: ("3 — Recap & Analysis", False),
            }
            text, done = labels[x]
            return f"✓ {text}" if done else f"  {text}"

        st.sidebar.divider()
        phase = st.sidebar.radio(
            "Navigate",
            options=[1, 2, 3],
            format_func=_phase_label,
            index=st.session_state.current_phase - 1,
            key="sb_phase_nav",
        )
        st.session_state.current_phase = phase

        # Autosave indicator
        if st.session_state._last_saved:
            st.sidebar.caption(f"Autosaved at {st.session_state._last_saved}")
    else:
        st.sidebar.info("Create a new run or load an existing one to begin.")


# ---------------------------------------------------------------------------
# Phase 1 — Prompt Refraction
# ---------------------------------------------------------------------------

def render_phase1() -> None:
    st.header("Phase 1 — Prompt Refraction")
    cfg: RunConfig = st.session_state.run_config

    col_meta, col_proceed = st.columns([3, 1])
    with col_meta:
        st.markdown(f"**Seminal intention:** *{cfg.seminal_intention}*")
        st.caption(f"Run: `{cfg.run_id}` · Created: {cfg.created_at}")
    with col_proceed:
        if st.session_state.prompts:
            if st.button("Go to Phase 2 →", type="primary", use_container_width=True):
                st.session_state.current_phase = 2
                st.rerun()
    st.divider()

    # Lens editor
    with st.expander("Configure Lenses", expanded=False):
        st.caption("Edit lens names and descriptions before generating. Changes apply to next generation.")
        lens_data = cfg.lenses.copy()
        for i, lens in enumerate(lens_data):
            c1, c2 = st.columns([1, 3])
            with c1:
                lens["lens_name"] = st.text_input(
                    f"Lens {i+1} name",
                    value=lens["lens_name"],
                    key=f"ln_{i}",
                )
            with c2:
                lens["description"] = st.text_input(
                    "Description",
                    value=lens.get("description", ""),
                    key=f"ld_{i}",
                )
            lens_data[i] = lens
        if st.button("Save lens config"):
            cfg.lenses = lens_data
            st.session_state.run_config = cfg
            storage.save_config(cfg)
            st.success("Lenses saved.")

    # Generate button
    already_generated = len(st.session_state.prompts) > 0
    col_gen, col_regen = st.columns([2, 1])
    with col_gen:
        gen_label = "Regenerate 12 Prompts" if already_generated else "Generate 12 Prompts"
        gen_clicked = st.button(gen_label, type="primary")
    with col_regen:
        if already_generated:
            st.caption(f"{len(st.session_state.prompts)} prompts generated")

    if gen_clicked:
        with st.spinner("Refracting intention into 12 prompts…"):
            prompts = generate_prompts(cfg.seminal_intention, cfg.lenses)
            st.session_state.prompts = prompts
            storage.save_prompts(cfg.run_id, prompts)
        _clear_prompt_widget_cache(st.session_state.prompts)
        st.success("12 prompts generated and saved.")
        st.rerun()

    if not st.session_state.prompts:
        st.info("Click **Generate 12 Prompts** to begin.")
        return

    # Display prompts grouped by lens
    prompts: List[Prompt] = st.session_state.prompts
    lens_groups: dict[str, List[Prompt]] = {}
    for p in prompts:
        lens_groups.setdefault(p.lens_name, []).append(p)

    spec_badge = {"low": "🟢 Low", "medium": "🟡 Medium", "high": "🔴 High"}

    for lens_name, lens_prompts in lens_groups.items():
        st.subheader(f"⬡ {lens_name}")
        for p in lens_prompts:
            _render_prompt_card(p, spec_badge)

    st.divider()
    if st.button("Proceed to Phase 2 →", type="primary"):
        st.session_state.current_phase = 2
        st.rerun()


def _render_prompt_card(p: Prompt, spec_badge: dict) -> None:
    excluded_style = "~~" if p.excluded else ""
    locked_icon = "🔒 " if p.locked else ""
    badge = spec_badge.get(p.specificity, p.specificity)
    edited_flag = " ✏️" if p.edited_by_human else ""

    header = f"{locked_icon}{badge}{edited_flag}"
    if p.excluded:
        header += "  ~~excluded~~"

    with st.container(border=True):
        col_info, col_controls = st.columns([4, 1])
        with col_info:
            st.caption(header)
        with col_controls:
            # Lock toggle
            new_locked = st.checkbox(
                "Lock", value=p.locked, key=f"lock_{p.prompt_id}", help="Prevent editing"
            )
            if new_locked != p.locked:
                p.locked = new_locked
                _save_prompts()

            new_excl = st.checkbox(
                "Exclude", value=p.excluded, key=f"excl_{p.prompt_id}"
            )
            if new_excl != p.excluded:
                p.excluded = new_excl
                _save_prompts()

        # Editable text area
        if not p.locked:
            new_text = st.text_area(
                "Prompt text",
                value=p.text,
                height=130,
                key=f"text_{p.prompt_id}",
                label_visibility="collapsed",
            )
            if new_text != p.text:
                p.text = new_text
                p.word_count = len(new_text.split())
                p.edited_by_human = True
                _save_prompts()
        else:
            st.markdown(p.text)

        st.caption(f"`{p.prompt_id}` · {p.word_count} words")


# ---------------------------------------------------------------------------
# Phase 2 — Image Generation
# ---------------------------------------------------------------------------

def render_phase2() -> None:
    st.header("Phase 2 — Image Generation")

    if not st.session_state.prompts:
        st.warning("No prompts found — complete Phase 1 first.")
        if st.button("← Back to Phase 1"):
            st.session_state.current_phase = 1
            st.rerun()
        return

    included = _prompts_included()
    if not included:
        st.warning("All prompts are excluded. Re-include some in Phase 1.")
        if st.button("← Back to Phase 1"):
            st.session_state.current_phase = 1
            st.rerun()
        return

    target = len(included) * 4
    generated = _total_images_generated()
    remaining = target - generated

    # Header row: progress + proceed button
    col_prog, col_cta = st.columns([3, 1])
    with col_prog:
        progress_frac = generated / target if target > 0 else 0
        st.progress(progress_frac, text=f"Images generated: {generated} / {target}")
        st.caption("Simulated placeholder compositions — no AI model is called.")
    with col_cta:
        if generated == target:
            if st.button("Go to Phase 3 →", type="primary", use_container_width=True):
                st.session_state.current_phase = 3
                st.rerun()
        else:
            st.button(
                f"Phase 3 ({remaining} images remaining)",
                disabled=True,
                use_container_width=True,
                help="Generate all images to unlock Phase 3.",
            )

    st.divider()

    # Navigation toolbar: selector + prev/next + generate-all + pin count
    _render_phase2_toolbar(included, generated, target)

    st.divider()

    # Focused prompt — the one currently selected via the navigator
    idx = st.session_state.p2_focused_idx
    idx = max(0, min(idx, len(included) - 1))   # clamp if excluded list shrank
    st.session_state.p2_focused_idx = idx
    _render_focused_card(included[idx])

    st.divider()

    # Compact overview of all prompts so user can jump to any of them
    _render_prompt_overview(included)

    # Recurrence mode — continuous transformation layer
    _render_recurrence_section()


def _render_phase2_toolbar(included: List[Prompt], generated: int, target: int) -> None:
    """Selector + prev/next navigation, generate-all button, and pinned count."""
    n = len(included)
    idx = st.session_state.p2_focused_idx

    def _label(p: Prompt) -> str:
        done = len(_images_for_prompt(p.prompt_id)) == 4
        status = "✅" if done else "⬜"
        return f"{status} {p.prompt_id} · {p.lens_name} · {p.specificity.capitalize()}"

    options = [_label(p) for p in included]

    col_prev, col_sel, col_next, col_space, col_gen, col_pin = st.columns([1, 5, 1, 1, 4, 2])

    with col_prev:
        if st.button("←", disabled=idx == 0, key="p2_prev", help="Previous prompt"):
            st.session_state.p2_focused_idx = idx - 1
            st.rerun()

    with col_sel:
        # Clear stale selectbox cache when labels change (e.g. after image generation)
        cached = st.session_state.get("p2_selector")
        if cached is not None and cached not in options:
            del st.session_state["p2_selector"]
        chosen = st.selectbox(
            "Prompt",
            options=options,
            index=idx,
            key="p2_selector",
            label_visibility="collapsed",
        )
        try:
            new_idx = options.index(chosen)
        except ValueError:
            new_idx = idx
        if new_idx != idx:
            st.session_state.p2_focused_idx = new_idx
            st.rerun()

    with col_next:
        if st.button("→", disabled=idx == n - 1, key="p2_next", help="Next prompt"):
            st.session_state.p2_focused_idx = idx + 1
            st.rerun()

    with col_gen:
        remaining_count = sum(
            1 for p in included if len(_images_for_prompt(p.prompt_id)) < 4
        )
        if st.button(
            f"⚡ Generate remaining ({remaining_count})",
            type="primary",
            disabled=remaining_count == 0,
            key="p2_gen_all",
        ):
            _generate_all_remaining(included, st.session_state.prob_intervention)
            st.rerun()

    with col_pin:
        pinned = sum(1 for r in st.session_state.image_records if r.pinned)
        st.metric("Pinned", pinned)


def _render_focused_card(p: Prompt) -> None:
    """Full-detail card for the currently navigated prompt."""
    records = _images_for_prompt(p.prompt_id)
    is_done = len(records) == 4
    spec = p.specificity.capitalize()

    flags = []
    if p.locked:
        flags.append("🔒 Locked")
    if p.edited_by_human:
        flags.append("✏️ Edited")
    flag_str = "  " + "  ".join(flags) if flags else ""

    st.subheader(f"{p.lens_name} · {spec}{flag_str}")
    st.caption(f"`{p.prompt_id}` · {p.word_count} words")

    col_prompt, col_note = st.columns([3, 2])

    with col_prompt:
        st.caption("Prompt text — edit before generating (disabled once images exist unless you regenerate)")
        if p.locked:
            st.markdown(p.text)
        else:
            edited_text = st.text_area(
                "Prompt text",
                value=p.text,
                height=140,
                key=f"p2_text_{p.prompt_id}",
                label_visibility="collapsed",
                disabled=is_done,   # prevent silent edits after images are made
            )
            if not is_done and edited_text != p.text:
                p.text = edited_text
                p.word_count = len(edited_text.split())
                p.edited_by_human = True
                _save_prompts()

    with col_note:
        st.caption("Intervention note — shapes the image composition")
        note = st.text_area(
            "Intervention note",
            value=st.session_state.intervention_notes.get(p.prompt_id, ""),
            height=140,
            placeholder="Describe a shift, emphasis, or constraint…",
            key=f"intnote_{p.prompt_id}",
            label_visibility="collapsed",
        )
        st.session_state.intervention_notes[p.prompt_id] = note

    # Action row
    col_gen, col_regen, col_prob = st.columns([2, 2, 3])

    with col_gen:
        if not is_done:
            if st.button("Generate 4 Images", type="primary", key=f"gen_{p.prompt_id}"):
                with st.spinner("Generating…"):
                    _generate_for_prompt(p, note)
                st.rerun()
        else:
            st.success("4 / 4 generated")

    with col_regen:
        if is_done:
            if st.button("🔄 Regenerate All 4", key=f"regenall_{p.prompt_id}"):
                with st.spinner("Regenerating…"):
                    _generate_for_prompt(p, note)
                st.rerun()

    with col_prob:
        prob = st.toggle(
            "Probabilistic intervention (30%)",
            value=st.session_state.prob_intervention,
            key="p2_prob_toggle",
            help="When generating all remaining, ~30% of prompts are suggested for intervention.",
        )
        st.session_state.prob_intervention = prob

    # Image grid — always visible once generated, never collapses
    if records:
        st.divider()
        _render_image_grid(p, records)


def _render_prompt_overview(included: List[Prompt]) -> None:
    """Compact grid of all prompts grouped by lens. Click any to navigate."""
    with st.expander("All prompts — click to navigate", expanded=True):
        # Group by lens while preserving global index for navigation
        lens_groups: dict[str, list[tuple[int, Prompt]]] = {}
        for i, p in enumerate(included):
            lens_groups.setdefault(p.lens_name, []).append((i, p))

        for lens_name, items in lens_groups.items():
            st.caption(f"**{lens_name}**")
            cols = st.columns(len(items))
            for col, (i, p) in zip(cols, items):
                recs = _images_for_prompt(p.prompt_id)
                n_done = len(recs)
                pinned_here = sum(1 for r in recs if r.pinned)
                status = "✓" if n_done == 4 else f"{n_done}/4"
                pin_str = f" · {pinned_here} pinned" if pinned_here else ""
                is_focused = (i == st.session_state.p2_focused_idx)
                with col:
                    if st.button(
                        f"{status} {p.specificity[:3]}",
                        key=f"ovv_{p.prompt_id}",
                        type="primary" if is_focused else "secondary",
                        use_container_width=True,
                        help=f"{p.prompt_id} · {p.specificity.capitalize()}{pin_str}",
                    ):
                        st.session_state.p2_focused_idx = i
                        st.rerun()


def _generate_for_prompt(p: Prompt, note: str) -> None:
    """Generate 4 images for a prompt, replacing any existing records for it."""
    new_records = [_do_generate_images(p, v, note) for v in range(4)]
    st.session_state.image_records = [
        r for r in st.session_state.image_records
        if r.source_prompt_id != p.prompt_id
    ] + new_records
    _save_records()


def _generate_all_remaining(included: List[Prompt], prob_mode: bool) -> None:
    """Generate 4 images for every included prompt that has none yet."""
    import random

    generated_prompts = {r.source_prompt_id for r in st.session_state.image_records}
    remaining = [p for p in included if p.prompt_id not in generated_prompts]
    if not remaining:
        return

    if prob_mode:
        rng = random.Random(st.session_state.run_id)  # deterministic per run
        flagged = [p.prompt_id for p in remaining if rng.random() < 0.30]
        if flagged:
            st.toast(
                f"Probabilistic mode: {', '.join(flagged)} suggested for intervention — "
                "add notes in those cards then regenerate."
            )

    prog = st.progress(0.0, text="Starting…")
    for i, p in enumerate(remaining):
        note = st.session_state.intervention_notes.get(p.prompt_id, "")
        new_records = [_do_generate_images(p, v, note) for v in range(4)]
        st.session_state.image_records.extend(new_records)
        _save_records()
        prog.progress((i + 1) / len(remaining), text=f"Generated {p.prompt_id} ({i+1}/{len(remaining)})")
    prog.empty()


def _render_image_grid(p: Prompt, records: List[ImageRecord]) -> None:
    """4-column image grid with pin, note, and single-variation regenerate."""
    # Sort by variation_index so they always display in order v1 v2 v3 v4
    records = sorted(records, key=lambda r: r.variation_index)
    cols = st.columns(4)
    for rec in records:
        with cols[rec.variation_index % 4]:
            img_path = Path(rec.image_path)
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"Missing: {img_path.name}")

            # Pin
            pinned = st.checkbox("📌 Pin", value=rec.pinned, key=f"pin_{rec.image_id}")
            if pinned != rec.pinned:
                rec.pinned = pinned
                _save_records()

            # User note
            saved_note = st.session_state.image_user_notes.get(rec.image_id, rec.user_note or "")
            new_note = st.text_input(
                "Note",
                value=saved_note,
                key=f"unote_{rec.image_id}",
                placeholder="Observation…",
                label_visibility="collapsed",
            )
            if new_note != saved_note:
                st.session_state.image_user_notes[rec.image_id] = new_note
                rec.user_note = new_note
                _save_records()

            # Regenerate this one variation
            if st.button("🔄", key=f"regen1_{rec.image_id}", help="Regenerate this variation"):
                current_note = st.session_state.intervention_notes.get(p.prompt_id, "")
                new_rec = _do_generate_images(p, rec.variation_index, current_note + "_regen")
                new_rec.pinned = rec.pinned
                new_rec.user_note = rec.user_note
                st.session_state.image_records = [
                    new_rec if r.image_id == rec.image_id else r
                    for r in st.session_state.image_records
                ]
                _save_records()
                st.rerun()

            st.caption(f"v{rec.variation_index + 1}")


# ---------------------------------------------------------------------------
# Phase 3 — Recap & Analysis
# ---------------------------------------------------------------------------

def render_phase3() -> None:
    st.header("Phase 3 — Recap & Analysis")

    prompts: List[Prompt] = st.session_state.prompts
    records: List[ImageRecord] = st.session_state.image_records
    cfg: RunConfig = st.session_state.run_config

    if not prompts:
        st.warning("No prompts found — complete Phase 1 first.")
        if st.button("← Back to Phase 1"):
            st.session_state.current_phase = 1
            st.rerun()
        return

    # Pre-compute summary counts used by multiple sections
    included = [p for p in prompts if not p.excluded]
    excluded_count = len(prompts) - len(included)
    edited_count = sum(1 for p in prompts if p.edited_by_human)
    locked_count = sum(1 for p in prompts if p.locked)
    pinned_records = [r for r in records if r.pinned]
    pinned_count = len(pinned_records)
    total_images = len(records)
    with_notes = sum(1 for r in records if r.user_note)

    # -------------------------------------------------------------------------
    # Section 1 — Run summary
    # -------------------------------------------------------------------------
    st.subheader("Run Summary")
    st.caption(
        f"**{cfg.run_id}** · {cfg.created_at} · "
        f"*{cfg.seminal_intention[:80]}{'…' if len(cfg.seminal_intention) > 80 else ''}*"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Prompts", len(prompts))
    c2.metric("Excluded", excluded_count)
    c3.metric("Edited", edited_count)
    c4.metric("Locked", locked_count)
    c5.metric("Images", total_images)
    c6.metric("Pinned", pinned_count)

    st.divider()

    # -------------------------------------------------------------------------
    # Section 2 — Vocabulary Analysis
    # Reads from session state on every render, so prompt edits are always
    # reflected immediately without any explicit refresh step.
    # -------------------------------------------------------------------------
    st.subheader("Vocabulary Analysis")
    st.caption(
        "Analyzes current prompt texts (including any human edits). "
        "Stopwords removed. Excluded prompts omitted."
    )

    include_int = st.toggle(
        "Include intervention notes",
        value=st.session_state.include_interventions_vocab,
        key="vocab_toggle",
        help="Folds intervention note text into the frequency counts.",
    )
    st.session_state.include_interventions_vocab = include_int

    vocab = analyze_vocab(
        prompts,
        image_records=records if include_int else None,
        include_interventions=include_int,
    )
    total_tokens = vocab["total_tokens"]

    recurrent_recs = [r for r in records if r.is_recurrent]
    tab_labels = ["Overall top 20", "By lens", "By specificity"]
    if recurrent_recs:
        tab_labels.append("Recurrence Evolution")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_vocab_overall(vocab, total_tokens)

    with tabs[1]:
        _render_vocab_by_lens(vocab, total_tokens)

    with tabs[2]:
        _render_vocab_by_specificity(vocab, total_tokens)

    if recurrent_recs:
        with tabs[3]:
            _render_vocab_recurrence(vocab, recurrent_recs, total_tokens)
            _render_provider_breakdown(recurrent_recs)

    st.divider()

    # -------------------------------------------------------------------------
    # Section 3 — Pinned image gallery
    # -------------------------------------------------------------------------
    st.subheader("Pinned Image Gallery")

    # Build a prompt lookup once
    prompt_by_id = {p.prompt_id: p for p in prompts}

    if not pinned_records:
        st.info("No images pinned yet. Go to Phase 2 to pin images.")
    else:
        st.caption(f"{pinned_count} pinned image{'s' if pinned_count != 1 else ''}")
        cols = st.columns(4)
        for i, rec in enumerate(pinned_records):
            with cols[i % 4]:
                img_path = Path(rec.image_path)
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.warning(f"File missing: {img_path.name}")
                    continue

                src = prompt_by_id.get(rec.source_prompt_id)
                if src:
                    st.caption(
                        f"**{src.lens_name}** · {src.specificity.capitalize()}  \n"
                        f"`{rec.image_id}`  v{rec.variation_index + 1}"
                    )
                    # Short prompt excerpt
                    excerpt = src.text[:120].rstrip()
                    if len(src.text) > 120:
                        excerpt += "…"
                    st.caption(f"*{excerpt}*")
                else:
                    st.caption(f"`{rec.image_id}`")

                if rec.user_note:
                    st.caption(f"Note: {rec.user_note}")
                if rec.intervention_note:
                    st.caption(f"Intervention: *{rec.intervention_note}*")

    st.divider()

    # -------------------------------------------------------------------------
    # Section 4 — Export (single-step: all data generated in memory)
    # -------------------------------------------------------------------------
    st.subheader("Export")
    st.caption("All files are generated from current session state — no intermediate save step needed.")

    _render_export_buttons(cfg, prompts, records, vocab, {
        "total_prompts": len(prompts),
        "excluded": excluded_count,
        "edited": edited_count,
        "locked": locked_count,
        "total_images": total_images,
        "pinned": pinned_count,
        "images_with_notes": with_notes,
    })

    # ── Section 5 — Recurrence Drift Analysis (only when recurrent data exists) ──
    recurrent_recs = [r for r in records if r.is_recurrent]
    if recurrent_recs:
        st.divider()
        st.subheader("Recurrence Drift Analysis")
        # Build provider summary for the caption
        from collections import Counter as _Counter
        provider_counts = _Counter(
            r.provider_name for r in recurrent_recs if r.provider_name
        )
        provider_summary = "  ·  ".join(
            f"`{p}` ×{n}" for p, n in provider_counts.most_common()
        ) or "`sim`"
        st.caption(
            f"{len(recurrent_recs)} recurrent image{'s' if len(recurrent_recs) != 1 else ''} "
            f"across {st.session_state.recurrence_iteration} iterations  ·  "
            f"providers: {provider_summary}  ·  "
            "Tracks how vocabulary transforms as the seminal intention metabolises over time."
        )
        drift = analyze_drift(prompts, recurrent_recs, cfg.seminal_intention)
        _render_drift_analysis(drift)


# ---------------------------------------------------------------------------
# Phase 3 helpers — vocab display
# ---------------------------------------------------------------------------

def _vocab_table(counter, n: int, total_tokens: int) -> pd.DataFrame:
    """Build a ranked DataFrame from a Counter for display."""
    rows = [
        {
            "rank": i + 1,
            "word": word,
            "count": count,
            "% of corpus": f"{count / total_tokens * 100:.1f}%",
        }
        for i, (word, count) in enumerate(counter.most_common(n))
    ]
    return pd.DataFrame(rows).set_index("rank")


def _render_vocab_overall(vocab: dict, total_tokens: int) -> None:
    overall = vocab["overall_freq"]
    if not overall:
        st.info("No vocabulary — generate prompts first.")
        return

    col_table, col_chart = st.columns([2, 3])

    with col_table:
        st.caption("Ranked word table")
        df = _vocab_table(overall, 20, total_tokens)
        st.dataframe(df, use_container_width=True, height=420)

    with col_chart:
        st.caption("Frequency distribution (top 20)")
        chart_data = pd.DataFrame(
            top_n(overall, 20), columns=["word", "count"]
        ).set_index("word")
        st.bar_chart(chart_data, height=420)


def _render_vocab_by_lens(vocab: dict, total_tokens: int) -> None:
    by_lens = vocab["by_lens"]
    if not by_lens:
        st.info("No lens data — generate prompts first.")
        return

    lenses = list(by_lens.keys())
    # Two lenses per row
    for row_start in range(0, len(lenses), 2):
        row_lenses = lenses[row_start: row_start + 2]
        cols = st.columns(len(row_lenses))
        for col, lens_name in zip(cols, row_lenses):
            counter = by_lens[lens_name]
            lens_total = sum(counter.values()) or 1
            with col:
                st.markdown(f"**{lens_name}**")
                df = _vocab_table(counter, 10, lens_total)
                st.dataframe(df, use_container_width=True, height=300)
        if row_start + 2 < len(lenses):
            st.divider()


def _render_vocab_by_specificity(vocab: dict, total_tokens: int) -> None:
    by_spec = vocab["by_specificity"]
    if not by_spec:
        st.info("No specificity data — generate prompts first.")
        return

    cols = st.columns(3)
    for col, spec in zip(cols, ["low", "medium", "high"]):
        counter = by_spec.get(spec)
        if not counter:
            continue
        spec_total = sum(counter.values()) or 1
        with col:
            st.markdown(f"**{spec.capitalize()} specificity**")
            df = _vocab_table(counter, 10, spec_total)
            st.dataframe(df, use_container_width=True, height=300)


def _render_provider_breakdown(recurrent_records: List[ImageRecord]) -> None:
    """
    Show a per-provider image count table beneath the vocabulary comparison.
    Logged for every recurrent image event via ImageRecord.provider_name.
    """
    from collections import Counter as _Counter
    counts = _Counter(
        r.provider_name or "unknown" for r in recurrent_records
    )
    if not counts:
        return

    st.divider()
    st.markdown("**Provider log**")
    st.caption("Images generated per provider across all recurrence iterations.")

    rows = [
        {
            "provider": name,
            "tier": provider_tier(name),
            "images": count,
            "pct": f"{count / len(recurrent_records) * 100:.1f}%",
        }
        for name, count in counts.most_common()
    ]
    st.dataframe(
        pd.DataFrame(rows).set_index("provider"),
        use_container_width=True,
        height=min(60 + len(rows) * 35, 280),
    )


def _render_vocab_recurrence(
    vocab_original: dict,
    recurrent_records: List[ImageRecord],
    total_tokens_original: int,
) -> None:
    """Side-by-side comparison of original vs recurrent vocabulary."""
    from collections import Counter as _Counter
    recurrent_freq: _Counter = _Counter()
    seen: set[str] = set()
    for rec in recurrent_records:
        text = rec.current_prompt_text or rec.parent_prompt_text
        if text and text not in seen:
            from core.vocab import tokenize
            recurrent_freq.update(tokenize(text))
            seen.add(text)

    if not recurrent_freq:
        st.info("No recurrent prompt data yet.")
        return

    total_rec = sum(recurrent_freq.values()) or 1
    col_orig, col_rec = st.columns(2)

    with col_orig:
        st.markdown("**Original prompts — top 20**")
        df_orig = _vocab_table(vocab_original["overall_freq"], 20, total_tokens_original)
        st.dataframe(df_orig, use_container_width=True, height=420)

    with col_rec:
        st.markdown("**Recurrent mutations — top 20**")
        df_rec = _vocab_table(recurrent_freq, 20, total_rec)
        st.dataframe(df_rec, use_container_width=True, height=420)


def _render_drift_analysis(drift: dict) -> None:
    """
    Display anchor / emerging / fading terms and the drift-over-time chart.
    Called from render_phase3 only when recurrent records exist.
    """
    col_anchor, col_emerge, col_fade = st.columns(3)

    with col_anchor:
        st.markdown("**Anchor terms**")
        st.caption("From the seminal intention — still present in recurrent prompts")
        if drift["anchor_terms"]:
            st.write("  ·  ".join(drift["anchor_terms"]))
        else:
            st.caption("*(none detected — full drift from intention vocabulary)*")

    with col_emerge:
        st.markdown("**Emerging terms**")
        st.caption("New vocabulary appearing in recurrent mutations")
        if drift["emerging_terms"]:
            df_em = pd.DataFrame(drift["emerging_terms"], columns=["word", "count"])
            st.dataframe(df_em.set_index("word"), use_container_width=True, height=220)
        else:
            st.caption("*(none yet)*")

    with col_fade:
        st.markdown("**Fading terms**")
        st.caption("Original vocabulary significantly declining in recurrent prompts")
        if drift["fading_terms"]:
            st.write("  ·  ".join(drift["fading_terms"]))
        else:
            st.caption("*(vocabulary holding stable)*")

    # Drift-over-time chart
    if drift["drift_over_time"]:
        st.markdown("**Similarity drift over iterations**")
        st.caption(
            "Average Jaccard similarity of each recurrent mutation to its original "
            "Phase-1 prompt text. 1.0 = unchanged; lower = further from origin."
        )
        drift_df = pd.DataFrame(drift["drift_over_time"]).set_index("iteration")
        st.line_chart(drift_df["avg_similarity"], height=220)
    else:
        st.caption("Drift chart will appear once multiple iterations have completed.")


# ---------------------------------------------------------------------------
# Phase 3 helpers — exports (all in-memory, single download step)
# ---------------------------------------------------------------------------

def _render_export_buttons(
    cfg: RunConfig,
    prompts: List[Prompt],
    records: List[ImageRecord],
    vocab: dict,
    metrics: dict,
) -> None:
    import csv
    import io

    run_id = cfg.run_id

    # Build all payloads in memory
    prompts_json = json.dumps([p.to_dict() for p in prompts], indent=2)

    image_log_json = json.dumps([r.to_dict() for r in records], indent=2)

    vocab_buf = io.StringIO()
    writer = csv.DictWriter(
        vocab_buf,
        fieldnames=["word", "count", "pct", "group", "group_value"],
    )
    writer.writeheader()
    writer.writerows(vocab["rows"])
    vocab_csv = vocab_buf.getvalue()

    recap = {
        "run_id": run_id,
        "seminal_intention": cfg.seminal_intention,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "top_words": [
            {"word": w, "count": c, "pct": round(c / vocab["total_tokens"] * 100, 2)}
            for w, c in top_n(vocab["overall_freq"], 20)
        ],
        "pinned_images": [r.to_dict() for r in records if r.pinned],
    }
    recap_json = json.dumps(recap, indent=2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            "⬇ prompts.json",
            data=prompts_json.encode(),
            file_name=f"{run_id}_prompts.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption(f"{len(prompts)} prompts")

    with col2:
        st.download_button(
            "⬇ image_log.json",
            data=image_log_json.encode(),
            file_name=f"{run_id}_image_log.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption(f"{len(records)} records")

    with col3:
        st.download_button(
            "⬇ vocab.csv",
            data=vocab_csv.encode(),
            file_name=f"{run_id}_vocab.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(vocab['rows'])} rows · overall + per-lens + per-spec")

    with col4:
        st.download_button(
            "⬇ recap.json",
            data=recap_json.encode(),
            file_name=f"{run_id}_recap.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption(f"Top words + {metrics['pinned']} pinned images")


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------

def render_welcome() -> None:
    st.title("⬡ Metabolic Prompt Studio")
    st.info(
        "**Demo mode** — all prompt generation and image outputs are simulated locally. "
        "No external API calls are made.",
        icon="ℹ️",
    )
    st.markdown(
        """
A concept-demo workflow for architectural AI prompt design.

**Three phases:**
1. **Prompt Refraction** — Enter a seminal intention. The app refracts it into 12 prompts across 4 conceptual lenses × 3 specificity levels.
2. **Image Generation** — Each prompt generates 4 placeholder image variations. You intervene, evaluate, and pin.
3. **Recap & Analysis** — Vocabulary frequency analysis, pinned gallery, and data export.

**To begin:** enter a seminal intention in the sidebar and click *Create Run* →
"""
    )


# ---------------------------------------------------------------------------
# Main routing
# ---------------------------------------------------------------------------

render_sidebar()

if st.session_state.run_id is None:
    render_welcome()
else:
    phase = st.session_state.current_phase
    if phase == 1:
        render_phase1()
    elif phase == 2:
        render_phase2()
    elif phase == 3:
        render_phase3()
