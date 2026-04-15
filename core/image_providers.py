"""
Image provider adapter layer — Metabolic Prompt Studio.

Each provider implements ImageProvider.generate() and returns a normalized
ImageResult. core/recurrence.py works through this interface exclusively
and never imports any provider-specific SDK directly.

Built-in providers
------------------
  sim                 Always available. Deterministic geometric placeholder (image_sim.py).
                      No API key required.

  gemini_flash_image  Google Imagen 3 Fast via google-genai SDK.
                      Fastest Imagen variant; good quality, lower cost.
                      Env: GOOGLE_API_KEY

  gemini_pro_image    Google Imagen 3 (standard) via google-genai SDK.
                      Highest Imagen quality; best for curated architectural renders.
                      Env: GOOGLE_API_KEY

  openai_gpt_image    OpenAI DALL-E 3 via openai SDK.
                      Strongest instruction-following for complex spatial language.
                      Env: OPENAI_API_KEY

  replicate_flux      Black Forest Labs Flux Schnell via replicate SDK.
                      Fastest live provider; lowest cost per image.
                      Env: REPLICATE_API_TOKEN

Adding a provider
-----------------
  1. Subclass ImageProvider and implement name, model, and generate().
  2. Call register(YourProvider()) at module level.
  3. Select it by string name with get_provider("your_name").
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Normalized result type
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    """
    Unified return object from any image provider.
    recurrence.py reads these fields to build the ImageRecord lineage entry.

    prompt_used   — the exact string sent to the API (may be shaped and/or
                    revised by the model, e.g. DALL-E 3 revised_prompt).
    raw_prompt    — the unshaped recurrent text before provider formatting.
                    Equals prompt_used for sim; differs for live providers.
    """
    image_path:         str               # relative path where the file was saved
    provider_name:      str               # e.g. "gemini_flash_image"
    model_name:         str               # specific model/version used
    prompt_used:        str               # shaped + possibly model-revised prompt
    raw_prompt:         str = ""          # unshaped recurrent text (lineage anchor)
    generation_time_ms: int = 0           # wall-clock generation time in milliseconds
    metadata:           Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ProviderError(RuntimeError):
    """
    Raised when a provider cannot complete generation.
    Covers: missing SDK, missing API key, quota exceeded, API error.
    recurrence.py catches this and falls back to SimProvider.
    """


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ImageProvider(ABC):
    """
    Contract for all image generation backends.
    Implementations should be stateless across calls.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in config and UI, e.g. 'gemini_flash_image'."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Full model/version string, e.g. 'imagen-3.0-fast-generate-001'."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_path: Path,
        variation_index: int = 0,
        **kwargs: Any,
    ) -> ImageResult:
        """
        Generate one image from prompt and write it to output_path.

        recurrence.py always passes these kwargs (providers may ignore them):
          lens_id (int), lens_name (str), specificity (str),
          intervention_note (str)
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, ImageProvider] = {}


def register(provider: ImageProvider) -> None:
    _REGISTRY[provider.name] = provider


def get_provider(name: str) -> ImageProvider:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY) or "(none registered)"
        raise ProviderError(
            f"Unknown provider '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def available_providers() -> List[str]:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Prompt shaping
# ---------------------------------------------------------------------------
# Each provider responds best to a different prompt style:
#
#   sim                 Pass through unchanged — no rendering model to guide.
#
#   replicate_flux      Flux models are trained on short, dense image-tag
#                       notation. Long discursive sentences degrade output.
#                       Shape: extract noun phrases and material/quality terms,
#                       join as a comma-separated tag string, append style
#                       suffixes for photorealistic architectural output.
#
#   gemini_flash_image  Imagen models parse natural-language scene descriptions
#                       well but are sensitive to ambiguity and abstraction.
#                       Shape: rewrite as a single clear scene sentence with
#                       explicit subject/setting, material qualities, lighting
#                       condition, and camera framing. Cap at ~200 tokens.
#
#   openai_gpt_image    DALL-E 3 has the strongest instruction-following of
#                       any provider. Pass the full architectural text with a
#                       rendering instruction prefix and explicit style anchors.
#                       The model handles complex spatial language natively.
#
# The raw recurrent prompt is always preserved in ImageResult.metadata so
# lineage tracking is never broken by shaping.

import re as _re


def _extract_noun_phrases(text: str) -> List[str]:
    """
    Lightweight heuristic extraction of noun-like phrases for Flux tag notation.
    Keeps capitalized compounds, hyphenated adjective-nouns, and domain terms.
    Not a full NLP parse — good enough for architectural vocabulary.
    """
    # Split on sentence boundaries, then on commas/semicolons
    chunks = _re.split(r"[.;,]", text)
    tags: List[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Take the first 6 words of each clause as a candidate tag
        words = chunk.split()
        phrase = " ".join(words[:6]).rstrip(".,;:")
        if len(phrase) > 4:
            tags.append(phrase)
    return tags[:12]   # cap at 12 tags to stay under Flux's preferred range


def shape_prompt(raw: str, provider_name: str, **kwargs: Any) -> str:
    """
    Return a provider-optimised version of the raw recurrent prompt text.
    Conceptual content and lineage vocabulary are preserved; only the
    surface form changes to match what each model responds to best.

    Args:
        raw:           The unshaped text produced by the recurrence mutation engine.
        provider_name: One of the registered provider names.
        **kwargs:      Optional context: specificity (str), lens_name (str).
                       Used to sharpen the scene framing for image providers.

    Returns:
        The shaped prompt string. Identical to raw for 'sim' and unknown providers.
    """
    if provider_name == "sim":
        return raw

    specificity = kwargs.get("specificity", "medium")
    lens_name   = kwargs.get("lens_name", "architectural")

    # ── replicate_flux: tag-notation ─────────────────────────────────────────
    if provider_name in ("replicate_flux",):
        tags = _extract_noun_phrases(raw)
        quality_suffix = (
            "photorealistic architectural render, sharp detail, natural light"
            if specificity == "high"
            else "architectural photography, ambient light, high detail"
            if specificity == "medium"
            else "architectural concept, soft light, minimal"
        )
        shaped = ", ".join(tags) + ", " + quality_suffix
        return shaped[:800]    # Flux effective prompt limit

    # ── gemini_flash_image: single clear scene sentence ──────────────────────
    if provider_name in ("gemini_flash_image", "gemini_pro_image"):
        # Strip the raw text to at most two sentences; prepend scene framing
        sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        core = " ".join(sentences[:2])
        # Cap length — Imagen degrades beyond ~180 tokens
        if len(core) > 900:
            core = core[:900].rsplit(" ", 1)[0]
        scene_prefix = f"Architectural scene — {lens_name} perspective: "
        lighting = (
            "dramatic directional light, photorealistic detail."
            if specificity == "high"
            else "soft natural light, detailed."
            if specificity == "medium"
            else "even ambient light, clean composition."
        )
        shaped = scene_prefix + core + " " + lighting
        return shaped

    # ── openai_gpt_image: full text with rendering instruction prefix ─────────
    if provider_name == "openai_gpt_image":
        render_prefix = (
            "Render as a high-fidelity architectural photograph. "
            "Photorealistic, no text, no watermarks. "
        )
        style_suffix = (
            " Ultra-detailed materiality. Professional architectural photography."
            if specificity == "high"
            else " Strong sense of space and light. Architectural quality."
            if specificity == "medium"
            else " Clean composition, legible spatial concept."
        )
        # DALL-E 3 hard limit is 4000 chars; leave room for affixes
        core = raw[:3800].rsplit(" ", 1)[0] if len(raw) > 3800 else raw
        shaped = render_prefix + core + style_suffix
        return shaped[:4000]

    # Unknown provider — pass through unchanged
    return raw


# ---------------------------------------------------------------------------
# sim — deterministic placeholder, always available
# ---------------------------------------------------------------------------

class SimProvider(ImageProvider):
    """
    Wraps core/image_sim.py. No API key or network required.
    Visual style is differentiated by lens; specificity controls density.
    Use during local development and as the guaranteed fallback.

    Cost:    Free
    Speed:   ~5–50 ms
    Quality: Geometric placeholder (no real architecture)
    """

    @property
    def name(self) -> str:
        return "sim"

    @property
    def model(self) -> str:
        return "image_sim/deterministic-v1"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        variation_index: int = 0,
        **kwargs: Any,
    ) -> ImageResult:
        from core.image_sim import generate_placeholder_image

        t0 = time.monotonic()
        generate_placeholder_image(
            prompt_text=prompt,
            variation_index=variation_index,
            intervention_note=kwargs.get("intervention_note", ""),
            lens_id=kwargs.get("lens_id", 1),
            lens_name=kwargs.get("lens_name", ""),
            specificity=kwargs.get("specificity", "medium"),
            output_path=output_path,
        )
        return ImageResult(
            image_path=str(output_path),
            provider_name=self.name,
            model_name=self.model,
            prompt_used=prompt,
            generation_time_ms=int((time.monotonic() - t0) * 1000),
        )


# ---------------------------------------------------------------------------
# gemini_flash_image — Google Imagen 3 Fast
# ---------------------------------------------------------------------------

class GeminiFlashImageProvider(ImageProvider):
    """
    Google Imagen 3 Fast via the google-genai SDK.

    Best for: high-volume recurring research runs where speed matters more
    than maximum quality. Produces genuine architectural imagery.

    Requirements:
        pip install google-genai
        GOOGLE_API_KEY environment variable

    Cost:    ~$0.02-0.04/image (verify at ai.google.dev/pricing)
    Speed:   ~3-6 seconds per image
    Quality: Good — photorealistic, strong spatial understanding
    """

    @property
    def name(self) -> str:
        return "gemini_flash_image"

    @property
    def model(self) -> str:
        return "imagen-3.0-fast-generate-001"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        variation_index: int = 0,
        **kwargs: Any,
    ) -> ImageResult:
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError:
            raise ProviderError(
                "google-genai SDK not installed. "
                "Run: pip install google-genai"
            )

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError(
                "GOOGLE_API_KEY environment variable is not set."
            )

        shaped = shape_prompt(prompt, self.name, **kwargs)
        client = genai.Client(api_key=api_key)
        t0 = time.monotonic()

        response = client.models.generate_images(
            model=self.model,
            prompt=shaped,
            config=gtypes.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1",
                output_mime_type="image/png",
            ),
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        image_bytes = response.generated_images[0].image.image_bytes
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        return ImageResult(
            image_path=str(output_path),
            provider_name=self.name,
            model_name=self.model,
            prompt_used=shaped,
            raw_prompt=prompt,
            generation_time_ms=elapsed_ms,
            metadata={"raw_prompt": prompt},
        )


# ---------------------------------------------------------------------------
# gemini_pro_image — Google Imagen 3
# ---------------------------------------------------------------------------

class GeminiProImageProvider(ImageProvider):
    """
    Google Imagen 3 (standard) via the google-genai SDK.

    Best for: curated runs where photorealistic quality and material detail
    matter. Stronger rendering of architectural surfaces and lighting.

    Requirements:
        pip install google-genai
        GOOGLE_API_KEY environment variable

    Cost:    ~$0.04-0.08/image (verify at ai.google.dev/pricing)
    Speed:   ~5-10 seconds per image
    Quality: Excellent — highest Imagen quality tier
    """

    @property
    def name(self) -> str:
        return "gemini_pro_image"

    @property
    def model(self) -> str:
        return "imagen-3.0-generate-001"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        variation_index: int = 0,
        **kwargs: Any,
    ) -> ImageResult:
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError:
            raise ProviderError(
                "google-genai SDK not installed. "
                "Run: pip install google-genai"
            )

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError(
                "GOOGLE_API_KEY environment variable is not set."
            )

        shaped = shape_prompt(prompt, self.name, **kwargs)
        client = genai.Client(api_key=api_key)
        t0 = time.monotonic()

        response = client.models.generate_images(
            model=self.model,
            prompt=shaped,
            config=gtypes.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1",
                output_mime_type="image/png",
            ),
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        image_bytes = response.generated_images[0].image.image_bytes
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        return ImageResult(
            image_path=str(output_path),
            provider_name=self.name,
            model_name=self.model,
            prompt_used=shaped,
            raw_prompt=prompt,
            generation_time_ms=elapsed_ms,
            metadata={"raw_prompt": prompt},
        )


# ---------------------------------------------------------------------------
# openai_gpt_image — OpenAI DALL-E 3
# ---------------------------------------------------------------------------

class OpenAIGPTImageProvider(ImageProvider):
    """
    OpenAI DALL-E 3 via the openai SDK.

    Best for: high-quality curated runs where the nuance of the mutated
    architectural prompt text matters most. DALL-E 3 has the strongest
    instruction-following of any widely available image model — it renders
    spatial relationships, material qualities, and atmospheric conditions
    described in complex text more faithfully than other providers.

    Note: DALL-E 3 may revise the prompt. The revised version is stored
    in ImageResult.prompt_used; the original is in ImageResult.metadata.

    Requirements:
        pip install openai
        OPENAI_API_KEY environment variable

    Cost:    ~$0.04/image standard, $0.08/image HD
    Speed:   ~8-15 seconds per image
    Quality: Excellent instruction-following, strong spatial coherence
    """

    @property
    def name(self) -> str:
        return "openai_gpt_image"

    @property
    def model(self) -> str:
        return "dall-e-3"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        variation_index: int = 0,
        **kwargs: Any,
    ) -> ImageResult:
        try:
            import openai as _openai
        except ImportError:
            raise ProviderError(
                "openai SDK not installed. "
                "Run: pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError(
                "OPENAI_API_KEY environment variable is not set."
            )

        import base64

        shaped  = shape_prompt(prompt, self.name, **kwargs)
        client  = _openai.OpenAI(api_key=api_key)
        t0      = time.monotonic()

        response = client.images.generate(
            model=self.model,
            prompt=shaped,                # already capped to 4000 chars by shape_prompt
            size="1024x1024",
            quality="standard",
            response_format="b64_json",
            n=1,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        image_bytes = base64.b64decode(response.data[0].b64_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)

        # DALL-E 3 often revises the prompt; capture what was actually used
        revised = getattr(response.data[0], "revised_prompt", shaped)

        return ImageResult(
            image_path=str(output_path),
            provider_name=self.name,
            model_name=self.model,
            prompt_used=revised,
            raw_prompt=prompt,
            generation_time_ms=elapsed_ms,
            metadata={"shaped_prompt": shaped, "raw_prompt": prompt},
        )


# ---------------------------------------------------------------------------
# replicate_flux — Black Forest Labs Flux Schnell
# ---------------------------------------------------------------------------

class ReplicateFluxProvider(ImageProvider):
    """
    Flux Schnell by Black Forest Labs via the Replicate API.

    Best for: high-volume research runs where cost is the primary constraint.
    Fastest live provider; excellent quality-to-cost ratio. Good for
    establishing visual patterns across many mutations before committing
    to a higher-cost provider for final outputs.

    Requirements:
        pip install replicate
        REPLICATE_API_TOKEN environment variable

    Cost:    ~$0.003/image (verify at replicate.com/pricing)
    Speed:   ~1-3 seconds per image
    Quality: Very good — strong general imagery, good architectural sense
    """

    @property
    def name(self) -> str:
        return "replicate_flux"

    @property
    def model(self) -> str:
        return "black-forest-labs/flux-schnell"

    def generate(
        self,
        prompt: str,
        output_path: Path,
        variation_index: int = 0,
        **kwargs: Any,
    ) -> ImageResult:
        try:
            import replicate as _replicate
        except ImportError:
            raise ProviderError(
                "replicate SDK not installed. "
                "Run: pip install replicate"
            )

        if not os.environ.get("REPLICATE_API_TOKEN"):
            raise ProviderError(
                "REPLICATE_API_TOKEN environment variable is not set."
            )

        import urllib.request

        shaped = shape_prompt(prompt, self.name, **kwargs)
        t0     = time.monotonic()
        output = _replicate.run(
            self.model,
            input={
                "prompt": shaped,
                "num_outputs": 1,
                "output_format": "png",
                "num_inference_steps": 4,  # Flux Schnell sweet spot
            },
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # output is a list of FileOutput objects; retrieve the first
        image_url = str(output[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(image_url, str(output_path))

        return ImageResult(
            image_path=str(output_path),
            provider_name=self.name,
            model_name=self.model,
            prompt_used=shaped,
            raw_prompt=prompt,
            generation_time_ms=elapsed_ms,
            metadata={"raw_prompt": prompt, "source_url": image_url},
        )


# ---------------------------------------------------------------------------
# Register all built-in providers
# ---------------------------------------------------------------------------

register(SimProvider())
register(GeminiFlashImageProvider())
register(GeminiProImageProvider())
register(OpenAIGPTImageProvider())
register(ReplicateFluxProvider())


# ---------------------------------------------------------------------------
# Rollout order — the four tiers exposed in the research UI
# ---------------------------------------------------------------------------

# Ordered from default/free to highest quality.
# gemini_pro_image is registered but intentionally excluded from the rollout
# selector; it remains callable via get_provider("gemini_pro_image") for
# scripted or advanced use, but the UI guides users through these four tiers.
ROLLOUT_PROVIDERS: List[str] = [
    "sim",
    "replicate_flux",
    "gemini_flash_image",
    "openai_gpt_image",
]

_PROVIDER_LABELS: Dict[str, str] = {
    "sim":               "sim — placeholder (free, instant)",
    "replicate_flux":    "replicate_flux — Flux Schnell (~$0.003/image, ~1-3 s)",
    "gemini_flash_image":"gemini_flash_image — Imagen 3 Fast (~$0.02-0.04/image, ~3-6 s)",
    "openai_gpt_image":  "openai_gpt_image — DALL-E 3, curated (~$0.04/image, ~8-15 s)",
}

_PROVIDER_TIERS: Dict[str, str] = {
    "sim":               "default",
    "replicate_flux":    "research",
    "gemini_flash_image":"comparison",
    "openai_gpt_image":  "curated",
}


def provider_label(name: str) -> str:
    """Human-readable label for the UI selectbox."""
    return _PROVIDER_LABELS.get(name, name)


def provider_tier(name: str) -> str:
    """Rollout tier for a provider name: default | research | comparison | curated."""
    return _PROVIDER_TIERS.get(name, "advanced")
