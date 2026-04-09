from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError as exc:  # pragma: no cover - helper script
    raise SystemExit(
        "Pillow is required for icon generation. Install it with: python -m pip install pillow"
    ) from exc


DEFAULT_BG = (2, 6, 23, 255)
CYAN = (56, 189, 248, 255)
MINT = (167, 243, 208, 255)
FOG = (226, 232, 240, 255)


def _draw_gradient_bg(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), DEFAULT_BG)
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            dx = (x - size / 2) / float(size)
            dy = (y - size / 2) / float(size)
            radial = max(0.0, 1.0 - math.sqrt(dx * dx + dy * dy) * 1.4)
            glow = max(0.0, 1.0 - abs(dx - 0.35) * 3.0 - abs(dy + 0.1) * 2.6)
            mix = min(1.0, radial * 0.42 + glow * 0.28)
            r = int(DEFAULT_BG[0] * (1 - mix) + 10 * mix)
            g = int(DEFAULT_BG[1] * (1 - mix) + 16 * mix)
            b = int(DEFAULT_BG[2] * (1 - mix) + 31 * mix)
            pixels[x, y] = (r, g, b, 255)
    return image.filter(ImageFilter.GaussianBlur(radius=max(1, size // 120)))


def _draw_network(image: Image.Image, *, maskable: bool) -> None:
    size = image.size[0]
    draw = ImageDraw.Draw(image)

    pad = int(size * (0.14 if maskable else 0.10))
    left = pad
    right = size - pad
    top = pad + int(size * 0.08)
    bottom = size - pad - int(size * 0.05)

    main_points = [
        (left + int(size * 0.04), bottom - int(size * 0.18)),
        (int(size * 0.30), int(size * 0.52)),
        (int(size * 0.50), int(size * 0.40)),
        (int(size * 0.68), int(size * 0.45)),
        (right - int(size * 0.07), bottom - int(size * 0.15)),
    ]
    alt_points = [
        (int(size * 0.22), int(size * 0.43)),
        (int(size * 0.42), int(size * 0.34)),
        (int(size * 0.60), int(size * 0.34)),
    ]

    draw.line(main_points, fill=CYAN, width=max(10, size // 18), joint="curve")
    draw.line(alt_points, fill=MINT, width=max(6, size // 28), joint="curve")

    for idx, (x, y) in enumerate(main_points):
        radius = int(size * (0.040 if idx in {0, len(main_points) - 1} else 0.032))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=FOG)

    core_x = int(size * 0.47)
    core_y = int(size * 0.55)
    core_r = int(size * 0.085)
    draw.ellipse((core_x - core_r, core_y - core_r, core_x + core_r, core_y + core_r), fill=MINT)
    inner_r = int(core_r * 0.52)
    draw.ellipse((core_x - inner_r, core_y - inner_r, core_x + inner_r, core_y + inner_r), fill=FOG)

    for offset in [0.0, 0.28, 0.56]:
        ring_r = int(size * (0.16 + offset * 0.03))
        alpha = int(120 - offset * 50)
        ring = Image.new("RGBA", image.size, (0, 0, 0, 0))
        ring_draw = ImageDraw.Draw(ring)
        ring_draw.ellipse(
            (core_x - ring_r, core_y - ring_r, core_x + ring_r, core_y + ring_r),
            outline=(56, 189, 248, max(35, alpha)),
            width=max(2, size // 64),
        )
        image.alpha_composite(ring)

    if maskable:
        clip_pad = int(size * 0.10)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            (clip_pad, clip_pad, size - clip_pad, size - clip_pad),
            radius=int(size * 0.18),
            outline=(226, 232, 240, 80),
            width=max(2, size // 120),
        )
        image.alpha_composite(overlay)


def make_icon(size: int, maskable: bool) -> Image.Image:
    image = _draw_gradient_bg(size)
    _draw_network(image, maskable=maskable)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Myco Android TWA launcher icons")
    parser.add_argument("--out-dir", default="static/twa-icons", help="Output directory for generated icons")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [192, 512]
    for size in sizes:
        make_icon(size, maskable=False).save(out_dir / f"mycelium-{size}.png")
        make_icon(size, maskable=True).save(out_dir / f"mycelium-{size}-maskable.png")

    print(f"Wrote {len(sizes) * 2} icons to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())