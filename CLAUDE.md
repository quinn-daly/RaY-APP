# CLAUDE.md — RaY-APP (Metabolic Prompt Studio)

## Project Context
This is a consulting project for Ray, an architecture student completing his directed research final project. The research question is: **how does AI understand and interpret architectural space?**

The app is a structured workflow that helps Ray run that research — taking a design intention, refracting it into prompts through different conceptual lenses, generating images from those prompts, and then analyzing the language and outputs to draw research conclusions.

This will be **presented as a final class project at the end of the semester.**

## Current Status — Prototype Phase
The app is fully functional as a prototype but uses **simulated outputs only** — no real API calls. This was intentional to validate the workflow design before committing to a live build.

**We are actively planning the transition from prototype to real build.** The core workflow, data models, and file structure are considered stable. The next major milestone is replacing the simulation modules with real API integrations.

## Stack
- **Python 3.10+**
- **Streamlit** — UI framework (runs as a local web app at `localhost:8501`)
- **Pillow** — currently used for placeholder image generation, will be replaced
- **pandas** — vocabulary analysis and data tables
- No database — all state saved as JSON files in `runs/`

## Project Structure
```
app.py              — main Streamlit app (all UI and phase logic)
requirements.txt    — pip dependencies
runs/               — auto-saved run data (do not manually edit)
  run_001/
    config.json     — seminal intention + lens settings
    prompts.json    — 12 prompts + any edits
    image_log.json  — image records, pins, notes
    images/         — generated PNG files
core/
  models.py         — dataclasses: RunConfig, Prompt, ImageRecord
  prompt_sim.py     — deterministic prompt generator (SIMULATED — to be replaced)
  image_sim.py      — placeholder image generator (SIMULATED — to be replaced)
  vocab.py          — vocabulary frequency analysis
  storage.py        — JSON read/write helpers for runs/
  __init__.py
```

## The Three Phases

### Phase 1 — Prompt Refraction
User enters a **seminal intention** (a sentence describing a design idea, e.g. *"The vertical inhabitation of collective memory"*). The app refracts this into **12 prompts** across 4 lenses × 3 specificity levels (low/medium/high). User can edit, lock, or exclude prompts before moving on.

### Phase 2 — Image Generation
Each prompt generates **4 image variations** (up to 48 images per run). User can add an **intervention note** per prompt to shift composition, pin images they want to keep, and add observation notes.

### Phase 3 — Recap & Analysis
Vocabulary frequency analysis broken down by lens and specificity level — reveals the language patterns shaping Ray's design thinking. Also shows a pinned image gallery and exports everything to JSON/CSV. **This phase has room to grow** as the research develops.

## The Four Lenses
| Lens | Purpose |
|---|---|
| Parsed Complexity | Breaks the idea into its structural components |
| Surfaced Assumptions | Surfaces the unstated premises the idea relies on |
| Multiple Perspectives | Views the idea through different stakeholders and disciplines |
| Logical Scaffolding | Rebuilds the underlying argument and decision structure |

## Key Data Models (`core/models.py`)
- `RunConfig` — run ID, timestamp, seminal intention, lens config
- `Prompt` — prompt text, lens, specificity, locked/excluded/edited flags
- `ImageRecord` — image path, source prompt, variation index, pin status, intervention note, user note

## Current Status — Live Build (as of 2026-04-16)

Both simulation modules have been replaced with live API integrations:

| Layer | Implementation | Provider | Env var |
|---|---|---|---|
| Prompt generation | `core/prompt_gen.py` | OpenAI gpt-4o-mini | `OPENAI_API_KEY` |
| Image generation | `core/image_providers.py` | Replicate Flux Schnell (default) | `REPLICATE_API_TOKEN` |

**Fallback behavior:** If an API key is absent or a call fails, the system silently degrades to the deterministic `prompt_sim` / `image_sim` placeholder path. No crash, no data loss.

**Project structure additions:**
```
core/
  prompt_gen.py     — LLM-based prompt generation (OpenAI gpt-4o-mini, with sim fallback)
  image_providers.py — provider adapter layer (sim, replicate_flux, gemini_flash_image, openai_gpt_image)
  recurrence.py     — mutation engine + lineage tracking
```

**Provider selector:** Visible in the sidebar when a run is active. Controls Phase 2 initial generation. Recurrence mode has its own separate selector in-UI.

**ImageRecord lineage fields** (added for recurrence + live providers):
- `parent_prompt_text`, `current_prompt_text`, `raw_prompt_text` — full text lineage
- `mutation_note`, `semantic_similarity`, `generation_iteration`, `is_recurrent`
- `provider_name`, `generation_time_ms`

## Dev Setup
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Important Notes
- All app state lives in `st.session_state` — Streamlit reruns the full script on every interaction, this is normal
- `runs/` grows with every session — clean up manually if it gets large
- `core/prompt_sim.py` is **still required** — `recurrence.py` imports `_BANKS` and `extract_concept` from it for mutation vocabulary; do not remove it
- When adding providers, subclass `ImageProvider` in `image_providers.py`, implement `generate()`, and call `register(YourProvider())` at module level
