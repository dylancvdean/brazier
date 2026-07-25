#!/usr/bin/env python3
"""Shape the Brazier artwork into a macOS application icon.

macOS icons are not square. Every app in the Dock is drawn inside the same
rounded superellipse — the "squircle" — inset from the edges of its canvas so
that icons line up at a consistent visual size. A full-bleed square reads as
oversized and out of place beside them, and nothing in the toolchain applies
the shape for you: `electron-builder` converts whatever PNG it is given
straight into `.icns`.

The geometry follows Apple's macOS icon grid: a 1024 pt canvas with the body
occupying the middle 824 pt. The corner is a superellipse rather than a
circular arc, which is what makes the curvature continuous.

Run with `uv`, so Pillow is not a repository dependency:

    uv run --with pillow apps/desktop/scripts/make-macos-icon.py \
        Brazier-logo.png apps/desktop/build/icon-mac.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

CANVAS = 1024
# Apple's grid: the icon body is 824 of 1024, leaving 100 on each side.
BODY = 824
# Exponent of the superellipse |x|^n + |y|^n = 1. Four is a rounded rectangle;
# Apple's shape sits near five, where the straight edge meets the corner
# without the curvature jump an arc produces.
EXPONENT = 5.0
# Mask supersampling, so the edge is smooth after downscaling.
SCALE = 4


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
        left = round(radius - half)
        right = round(radius + half)
        for x in range(left, right):
            pixels[x, y] = 255
    return mask.resize((size, size), Image.LANCZOS)


def build_icon(source: Path, destination: Path) -> None:
    artwork = Image.open(source).convert("RGBA")
    # Cover the body square, cropping the longer side rather than distorting.
    ratio = max(BODY / artwork.width, BODY / artwork.height)
    resized = artwork.resize(
        (max(1, round(artwork.width * ratio)), max(1, round(artwork.height * ratio))),
        Image.LANCZOS,
    )
    left = (resized.width - BODY) // 2
    top = (resized.height - BODY) // 2
    body = resized.crop((left, top, left + BODY, top + BODY))
    body.putalpha(squircle_mask(BODY, EXPONENT, SCALE))

    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    offset = (CANVAS - BODY) // 2
    canvas.paste(body, (offset, offset), body)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    print(f"wrote {destination} ({CANVAS}x{CANVAS}, body {BODY})")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    build_icon(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
