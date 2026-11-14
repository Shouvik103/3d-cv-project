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
    if not focal35:
        # Most phone JPEGs store focal length in the nested Exif IFD.
        focal35 = exif.get_ifd(34665).get(41989)
    if focal35:
        focal = float(focal35) / 36.0 * max(width, height)
        source = "exif_35mm"
    else:
        focal, source = 1.2 * max(width, height), "fallback_1.2_max_side"
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]]), source


def prepare(images: Path, output: Path, max_side: int = 1600, intrinsics: Path | None = None, calibration: Path | None = None) -> dict:
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    if images.resolve() == output.resolve():
        raise ValueError("Input and output directories must differ")
    calibrated = yaml.safe_load(calibration.read_text()) if calibration else None
    if calibration and intrinsics:
        raise ValueError("Choose per-image intrinsics or checkerboard calibration")
    supplied = yaml.safe_load(intrinsics.read_text())["images"] if intrinsics else {}
    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for source in sorted(path for path in images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            if calibrated:
                import cv2
                if calibrated['image_size'] != list(image.size):
                    raise ValueError(f'Calibration resolution does not match {source.name}')
                original_K = np.array(calibrated['K'], dtype=float)
                distortion = np.array(calibrated['distortion'], dtype=float)
                K, _ = cv2.getOptimalNewCameraMatrix(original_K, distortion, image.size, 0, image.size)
                image = Image.fromarray(cv2.undistort(np.array(image), original_K, distortion, None, K))
                source_name = 'checkerboard_undistorted'
            elif source.name in supplied:
                K = np.array(supplied[source.name]["K"], dtype=float)
                if supplied[source.name].get("image_size", list(image.size)) != list(image.size):
                    raise ValueError(f"Calibration size mismatch for {source.name}")
                source_name = supplied[source.name].get("source", "supplied")
            elif intrinsics:
                raise ValueError(f"Missing calibration for {source.name}")
            else:
                K, source_name = estimate_intrinsics(image)
            scale = min(1.0, max_side / max(image.size))
            size = tuple(round(v * scale) for v in image.size)
            if scale < 1:
                image = image.resize(size, Image.Resampling.LANCZOS)
                K[:2] *= scale
                K[2, 2] = 1.0
            name = f"{source.stem}.jpg"
            if name in records:
                raise ValueError(f"Duplicate output filename: {name}")
            image.save(output / name, quality=95)
            records[name] = {"K": K.tolist(), "image_size": list(image.size), "source": source_name}
    if not records:
        raise ValueError("No supported images found")
    payload = {"images": records}
    (output.parent / "intrinsics.yaml").write_text(yaml.safe_dump(payload, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--intrinsics", type=Path, help="Preserve calibrated K while resizing")
    parser.add_argument("--calibration", type=Path, help="Checkerboard calibration.yaml; undistort before resizing")
    parser.add_argument("--poses", type=Path, help="Optional ground-truth poses.json to rename and rescale with images")
    args = parser.parse_args()
    payload = prepare(args.images, args.out, args.max_side, args.intrinsics, args.calibration)
    if args.poses:
        import json
        old = json.loads(args.poses.read_text())
        names = {name: f"{Path(name).stem}.jpg" for name in old["image_names"] if f"{Path(name).stem}.jpg" in payload["images"]}
        converted = {"image_names": list(names.values()), "registered": [names[n] for n in old["registered"] if n in names], "K": {}, "R": {}, "t": {}, "image_size": {}}
        for name, dest in names.items():
            converted["K"][dest] = payload["images"][dest]["K"]
            converted["image_size"][dest] = payload["images"][dest]["image_size"]
            if name in old["registered"]:
                converted["R"][dest], converted["t"][dest] = old["R"][name], old["t"][name]
        (args.out.parent / "gt_poses.json").write_text(json.dumps(converted, indent=2))
    print(f"prepared {len(payload['images'])} images in {args.out}")


if __name__ == "__main__":
    main()
