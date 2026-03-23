"""
Deterministic placeholder image generator for Metabolic Prompt Studio.

Each image is a 512×512 PNG derived from:
  - prompt text
  - variation index
  - intervention note

The visual style is differentiated by lens:
  1) Parsed Complexity → grid / orthogonal composition
  2) Surfaced Assumptions → diagonal / layered composition
  3) Multiple Perspectives → radial / concentric composition
  4) Logical Scaffolding → linear / hierarchical composition

Specificity affects density: low = sparse, medium = moderate, high = dense.
Text overlay shows lens, specificity, variation, and hash snippet.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

WIDTH = HEIGHT = 512
FONT = None  # populated lazily


def _get_font(size: int = 14):
    global FONT
    # Try common Windows system fonts
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _hash_bytes(prompt_text: str, variation_index: int, intervention_note: str) -> bytes:
    key = f"{prompt_text}|{variation_index}|{intervention_note}"
    return hashlib.sha256(key.encode("utf-8")).digest()


def _bytes_to_hsl_color(b0: int, b1: int, b2: int, sat_range=(0.35, 0.75), val_range=(0.45, 0.85)) -> Tuple[int, int, int]:
    """Map 3 bytes to an RGB colour with constrained saturation and lightness."""
    import colorsys
    h = b0 / 255.0
    s = sat_range[0] + (b1 / 255.0) * (sat_range[1] - sat_range[0])
    v = val_range[0] + (b2 / 255.0) * (val_range[1] - val_range[0])
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _contrasting(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return black or white for readable text overlay."""
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


# ---------------------------------------------------------------------------
# Lens-specific drawing routines
# ---------------------------------------------------------------------------

def _draw_grid(draw: ImageDraw.Draw, seed: bytes, density: int) -> None:
    """Parsed Complexity — orthogonal grid of rectangles."""
    cols = rows = density + 3          # 4–7 divisions
    cw = WIDTH // cols
    rh = HEIGHT // rows
    bidx = 6
    for r in range(rows):
        for c in range(cols):
            b = seed[bidx % 32]
            bidx += 1
            if b > 90:                 # ~65% chance of filled cell
                color = _bytes_to_hsl_color(seed[(bidx) % 32], seed[(bidx + 1) % 32], seed[(bidx + 2) % 32])
                bidx += 3
                margin = int(seed[bidx % 32] / 255 * (cw // 4))
                bidx += 1
                x0 = c * cw + margin
                y0 = r * rh + margin
                x1 = x0 + cw - margin
                y1 = y0 + rh - margin
                draw.rectangle([x0, y0, x1, y1], fill=color)


def _draw_diagonal(draw: ImageDraw.Draw, seed: bytes, density: int) -> None:
    """Surfaced Assumptions — diagonal bands and layered polygons."""
    n_layers = density + 2
    bidx = 6
    for i in range(n_layers):
        b = seed[bidx % 32]
        bidx += 1
        color = _bytes_to_hsl_color(seed[bidx % 32], seed[(bidx + 1) % 32], seed[(bidx + 2) % 32])
        bidx += 3
        # diagonal band
        offset = int((b / 255.0) * WIDTH * 1.6)
        thickness = int(seed[bidx % 32] / 255.0 * 150) + 30
        bidx += 1
        pts = [
            (offset - thickness, 0),
            (offset + thickness, 0),
            (offset + thickness - HEIGHT, HEIGHT),
            (offset - thickness - HEIGHT, HEIGHT),
        ]
        draw.polygon(pts, fill=color)


def _draw_radial(draw: ImageDraw.Draw, seed: bytes, density: int) -> None:
    """Multiple Perspectives — concentric rings and radiating wedges."""
    cx, cy = WIDTH // 2, HEIGHT // 2
    n_rings = density + 2
    bidx = 6
    for i in range(n_rings):
        color = _bytes_to_hsl_color(seed[bidx % 32], seed[(bidx + 1) % 32], seed[(bidx + 2) % 32])
        bidx += 3
        radius = int((i + 1) / (n_rings + 1) * (WIDTH // 2 - 10))
        thick = int(seed[bidx % 32] / 255.0 * 40) + 8
        bidx += 1
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=color,
            width=thick,
        )
    # radiating lines
    n_lines = density * 3 + 4
    for i in range(n_lines):
        color = _bytes_to_hsl_color(seed[bidx % 32], seed[(bidx + 1) % 32], seed[(bidx + 2) % 32])
        bidx += 3
        angle = (i / n_lines) * 2 * math.pi
        length = int(seed[bidx % 32] / 255.0 * (WIDTH // 2 - 20)) + 20
        bidx += 1
        ex = cx + int(length * math.cos(angle))
        ey = cy + int(length * math.sin(angle))
        draw.line([cx, cy, ex, ey], fill=color, width=2)


def _draw_hierarchical(draw: ImageDraw.Draw, seed: bytes, density: int) -> None:
    """Logical Scaffolding — horizontal bands and tree-like connectors."""
    n_levels = density + 3
    band_h = HEIGHT // n_levels
    bidx = 6
    for level in range(n_levels):
        color = _bytes_to_hsl_color(seed[bidx % 32], seed[(bidx + 1) % 32], seed[(bidx + 2) % 32])
        bidx += 3
        y0 = level * band_h
        y1 = y0 + band_h
        # Full band or partial
        if seed[bidx % 32] > 80:
            w_frac = 0.3 + (seed[(bidx + 1) % 32] / 255.0) * 0.7
            bidx += 2
            x1 = int(WIDTH * w_frac)
            draw.rectangle([0, y0 + 4, x1, y1 - 4], fill=color)
        else:
            bidx += 2
        # connector lines
        n_nodes = seed[bidx % 32] % (density + 2) + 1
        bidx += 1
        spacing = WIDTH // (n_nodes + 1)
        for node in range(n_nodes):
            nx = (node + 1) * spacing
            draw.ellipse([nx - 5, y0 + band_h // 2 - 5, nx + 5, y0 + band_h // 2 + 5], fill=color)
            if level < n_levels - 1:
                draw.line([nx, y0 + band_h // 2, nx, y1 + band_h // 2], fill=color, width=2)


_LENS_DRAWERS = {
    1: _draw_grid,
    2: _draw_diagonal,
    3: _draw_radial,
    4: _draw_hierarchical,
}


# ---------------------------------------------------------------------------
# Specificity → density
# ---------------------------------------------------------------------------

_DENSITY = {"low": 1, "medium": 3, "high": 5}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_placeholder_image(
    prompt_text: str,
    variation_index: int,
    intervention_note: str,
    lens_id: int,
    lens_name: str,
    specificity: str,
    output_path: str | Path,
) -> None:
    """
    Generate and save a deterministic placeholder image.
    All visual properties are derived from the hash of the inputs.
    """
    seed = _hash_bytes(prompt_text, variation_index, intervention_note)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Background colour
    bg = _bytes_to_hsl_color(seed[0], seed[1], seed[2], sat_range=(0.05, 0.2), val_range=(0.88, 0.97))
    img = Image.new("RGB", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(img)

    # Lens composition
    density = _DENSITY.get(specificity, 3)
    drawer = _LENS_DRAWERS.get(lens_id, _draw_grid)
    drawer(draw, seed, density)

    # Subtle border
    border_color = _bytes_to_hsl_color(seed[3], seed[4], seed[5], sat_range=(0.4, 0.7), val_range=(0.3, 0.6))
    draw.rectangle([0, 0, WIDTH - 1, HEIGHT - 1], outline=border_color, width=3)

    # Text overlay
    font_sm = _get_font(12)
    font_lg = _get_font(16)
    hash_snippet = seed.hex()[:8].upper()
    text_color = _contrasting(bg)

    # Top bar (semi-transparent dark strip)
    draw.rectangle([0, 0, WIDTH, 36], fill=(0, 0, 0, 180))
    draw.text((8, 6), f"{lens_name}  ·  {specificity.upper()}", font=font_lg, fill=(240, 240, 240))

    # Bottom bar
    draw.rectangle([0, HEIGHT - 30, WIDTH, HEIGHT], fill=(0, 0, 0, 160))
    draw.text((8, HEIGHT - 22), f"v{variation_index + 1}", font=font_sm, fill=(200, 200, 200))
    draw.text((WIDTH - 72, HEIGHT - 22), f"#{hash_snippet}", font=font_sm, fill=(200, 200, 200))

    # Watermark
    draw.text((8, 44), "SIMULATED", font=font_sm, fill=(180, 180, 180))

    img.save(str(output_path), "PNG")
