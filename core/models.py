"""
Data models for Metabolic Prompt Studio.
All models are plain dataclasses that serialize to/from dicts for JSON storage.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class RunConfig:
    run_id: str
    created_at: str
    seminal_intention: str
    lenses: List[Dict[str, Any]]
    settings: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        return cls(
            run_id=d["run_id"],
            created_at=d["created_at"],
            seminal_intention=d["seminal_intention"],
            lenses=d["lenses"],
            settings=d.get("settings", {}),
        )


@dataclass
class Prompt:
    prompt_id: str
    cycle: int
    lens_id: int
    lens_name: str
    specificity: str          # "low" | "medium" | "high"
    text: str
    word_count: int
    locked: bool = False
    excluded: bool = False
    edited_by_human: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Prompt":
        return cls(
            prompt_id=d["prompt_id"],
            cycle=d["cycle"],
            lens_id=d["lens_id"],
            lens_name=d["lens_name"],
            specificity=d["specificity"],
            text=d["text"],
            word_count=d["word_count"],
            locked=d.get("locked", False),
            excluded=d.get("excluded", False),
            edited_by_human=d.get("edited_by_human", False),
        )


@dataclass
class ImageRecord:
    image_id: str
    source_prompt_id: str
    variation_index: int
    intervention_note: str
    image_path: str           # relative path from project root
    pinned: bool = False
    created_at: str = ""
    user_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=d["image_id"],
            source_prompt_id=d["source_prompt_id"],
            variation_index=d["variation_index"],
            intervention_note=d.get("intervention_note", ""),
            image_path=d["image_path"],
            pinned=d.get("pinned", False),
            created_at=d.get("created_at", ""),
            user_note=d.get("user_note"),
        )
