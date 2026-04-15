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
    """
    image_path:         str               # relative path where the file was saved
    provider_name:      str               # e.g. "gemini_flash_image"
    model_name:         str               # specific model/version used
    prompt_used:        str               # exact prompt sent (may be revised by API)
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

        client = genai.Client(api_key=api_key)
        t0 = time.monotonic()

        response = client.models.generate_images(
            model=self.model,
            prompt=prompt,
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
            prompt_used=prompt,
            generation_time_ms=elapsed_ms,
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

        client = genai.Client(api_key=api_key)
        t0 = time.monotonic()

        response = client.models.generate_images(
            model=self.model,
            prompt=prompt,
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
            prompt_used=prompt,
            generation_time_ms=elapsed_ms,
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

        client = _openai.OpenAI(api_key=api_key)
        t0 = time.monotonic()

        response = client.images.generate(
            model=self.model,
            prompt=prompt[:4000],         # DALL-E 3 hard limit
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
        revised = getattr(response.data[0], "revised_prompt", prompt)

        return ImageResult(
            image_path=str(output_path),
            provider_name=self.name,
            model_name=self.model,
            prompt_used=revised,
            generation_time_ms=elapsed_ms,
            metadata={"original_prompt": prompt},
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

        t0 = time.monotonic()
        output = _replicate.run(
            self.model,
            input={
                "prompt": prompt,
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
            prompt_used=prompt,
            generation_time_ms=elapsed_ms,
            metadata={"source_url": image_url},
        )


# ---------------------------------------------------------------------------
# Register all built-in providers
# ---------------------------------------------------------------------------

register(SimProvider())
register(GeminiFlashImageProvider())
register(GeminiProImageProvider())
register(OpenAIGPTImageProvider())
register(ReplicateFluxProvider())
