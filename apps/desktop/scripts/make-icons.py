#!/usr/bin/env python3
"""Derive the application icons from the Brazier artwork.

Three files come out of one master, and each exists for a different reason:

- `build/icon.png` — the square artwork, which is what Windows and Linux use.
- `build/icon-mac.png` — cut to the macOS squircle. The Dock draws every icon
  inside that shape, inset so they line up at a consistent visual size, and
  nothing in the toolchain applies it: `electron-builder` converts whatever PNG
  it is handed straight into `.icns`. A full-bleed square reads as oversized
  and square-cornered beside its neighbours.
- `src/renderer/src/assets/brazier-logo.png` — the sidebar mark, small enough
  not to ship a megabyte into the renderer bundle.

The derived files are committed so that building needs no Python: this script
is for changing the artwork, not for building. Run it from the repository root
after replacing the master, and commit what it writes.

    uv run --with pillow apps/desktop/scripts/make-icons.py

The macOS geometry follows Apple's icon grid: a 1024 pt canvas with the body
occupying the middle 824 pt, and a superellipse corner rather than a circular
arc, which is what makes the curvature continuous.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

MASTER = Path("assets/brazier-logo.png")
DESKTOP = Path("apps/desktop")

CANVAS = 1024
# Apple's grid: the icon body is 824 of 1024, leaving 100 on each side.
BODY = 824
# Exponent of the superellipse |x|^n + |y|^n = 1. Four is a rounded rectangle;
# Apple's shape sits near five, where the straight edge meets the corner
# without the curvature jump an arc produces.
EXPONENT = 5.0
# Mask supersampling, so the edge is smooth after downscaling.
SCALE = 4
# Sidebar mark, at twice its rendered size for high-density displays.
SIDEBAR = 256


def squircle_mask(size: int, exponent: float, scale: int) -> Image.Image:
    """An anti-aliased superellipse mask of `size` pixels."""
    high = size * scale
    radius = high / 2
    mask = Image.new("L", (high, high), 0)
    pixels = mask.load()
    assert pixels is not None
    for y in range(high):
        # Normalised distance of this row from the centre, in [0, 1].
        dy = abs((y + 0.5) - radius) / radius
        remaining = 1.0 - dy**exponent
        if remaining <= 0:
            continue
        # Solve for the row's half-width on the superellipse.
        half = remaining ** (1.0 / exponent) * radius
        for x in range(round(radius - half), round(radius + half)):
            pixels[x, y] = 255
    return mask.resize((size, size), Image.LANCZOS)


def square(artwork: Image.Image, size: int) -> Image.Image:
    """Centre-crop to a square and resize, rather than distorting."""
    ratio = max(size / artwork.width, size / artwork.height)
    resized = artwork.resize(
        (max(1, round(artwork.width * ratio)), max(1, round(artwork.height * ratio))),
        Image.LANCZOS,
    )
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def write(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    print(f"wrote {destination} ({image.width}x{image.height})")


def main() -> int:
    if not MASTER.is_file():
        print(f"missing {MASTER} — run from the repository root", file=sys.stderr)
        return 1
    artwork = Image.open(MASTER).convert("RGBA")

    write(square(artwork, CANVAS), DESKTOP / "build/icon.png")
    write(square(artwork, SIDEBAR), DESKTOP / "src/renderer/src/assets/brazier-logo.png")

    body = square(artwork, BODY)
    body.putalpha(squircle_mask(BODY, EXPONENT, SCALE))
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    offset = (CANVAS - BODY) // 2
    canvas.paste(body, (offset, offset), body)
    write(canvas, DESKTOP / "build/icon-mac.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
