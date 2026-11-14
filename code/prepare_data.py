"""Normalize orientation, resize images, and write per-image intrinsics metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def estimate_intrinsics(image: Image.Image) -> tuple[np.ndarray, str]:
    width, height = image.size
    exif = image.getexif()
    focal35 = exif.get(41989)  # FocalLengthIn35mmFilm
    if focal35:
        focal = float(focal35) / 36.0 * max(width, height)
        source = "exif_35mm"
    else:
        focal, source = 1.2 * max(width, height), "fallback_1.2_max_side"
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]]), source


def prepare(images: Path, output: Path, max_side: int = 1600) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for source in sorted(path for path in images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            K, source_name = estimate_intrinsics(image)
            scale = min(1.0, max_side / max(image.size))
            size = tuple(round(v * scale) for v in image.size)
            if scale < 1:
                image = image.resize(size, Image.Resampling.LANCZOS)
                K[:2] *= scale
                K[2, 2] = 1.0
            name = f"{source.stem}.jpg"
            image.save(output / name, quality=95)
            records[name] = {"K": K.tolist(), "image_size": list(image.size), "source": source_name}
    payload = {"images": records}
    (output.parent / "intrinsics.yaml").write_text(yaml.safe_dump(payload, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1600)
    args = parser.parse_args()
    payload = prepare(args.images, args.out, args.max_side)
    print(f"prepared {len(payload['images'])} images in {args.out}")


if __name__ == "__main__":
    main()
