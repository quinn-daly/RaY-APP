"""
Disk persistence for Metabolic Prompt Studio.

Each run lives at:
  runs/run_NNN/
    config.json
    prompts.json
    image_log.json
    images/
    vocab.csv
    recap.json
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from core.models import ImageRecord, Prompt, RunConfig

RUNS_DIR = Path("runs")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def get_images_dir(run_id: str) -> Path:
    return get_run_dir(run_id) / "images"


def _ensure_run_dirs(run_id: str) -> None:
    get_images_dir(run_id).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

def _next_run_id() -> str:
    RUNS_DIR.mkdir(exist_ok=True)
    existing = [
        d.name for d in RUNS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return f"run_{(max(nums) + 1 if nums else 1):03d}"


def create_run(
    seminal_intention: str,
    lenses: List[dict],
    settings: Optional[dict] = None,
) -> RunConfig:
    RUNS_DIR.mkdir(exist_ok=True)
    run_id = _next_run_id()
    cfg = RunConfig(
        run_id=run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        seminal_intention=seminal_intention,
        lenses=lenses,
        settings=settings or {},
    )
    _ensure_run_dirs(run_id)
    save_config(cfg)
    return cfg


def list_runs() -> List[str]:
    RUNS_DIR.mkdir(exist_ok=True)
    runs = sorted(
        d.name for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )
    return runs


def load_run(run_id: str) -> Tuple[RunConfig, List[Prompt], List[ImageRecord]]:
    run_dir = get_run_dir(run_id)
    cfg = RunConfig.from_dict(_load_json(run_dir / "config.json"))
    prompts = [
        Prompt.from_dict(p)
        for p in _load_json(run_dir / "prompts.json", default=[])
    ]
    records = [
        ImageRecord.from_dict(r)
        for r in _load_json(run_dir / "image_log.json", default=[])
    ]
    return cfg, prompts, records


# ---------------------------------------------------------------------------
# Savers
# ---------------------------------------------------------------------------

def save_config(cfg: RunConfig) -> None:
    _ensure_run_dirs(cfg.run_id)
    _save_json(get_run_dir(cfg.run_id) / "config.json", cfg.to_dict())


def save_prompts(run_id: str, prompts: List[Prompt]) -> None:
    _save_json(get_run_dir(run_id) / "prompts.json", [p.to_dict() for p in prompts])


def save_image_records(run_id: str, records: List[ImageRecord]) -> None:
    _save_json(
        get_run_dir(run_id) / "image_log.json",
        [r.to_dict() for r in records],
    )


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def export_vocab_csv(run_id: str, rows: List[dict]) -> Path:
    """rows: list of dicts with keys: word, count, lens, specificity"""
    out = get_run_dir(run_id) / "vocab.csv"
    if not rows:
        out.write_text("word,count,lens,specificity\n")
        return out
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def export_recap_json(run_id: str, data: dict) -> Path:
    out = get_run_dir(run_id) / "recap.json"
    _save_json(out, data)
    return out


def export_run_json(run_id: str, cfg: RunConfig, prompts: List[Prompt], records: List[ImageRecord]) -> Path:
    out = get_run_dir(run_id) / "full_export.json"
    payload = {
        "config": cfg.to_dict(),
        "prompts": [p.to_dict() for p in prompts],
        "image_records": [r.to_dict() for r in records],
    }
    _save_json(out, payload)
    return out


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
