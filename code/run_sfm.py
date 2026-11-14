"""Command-line incremental sparse SfM reconstruction."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from code.sfm.bundle_adjust import bundle_adjust
from code.sfm.features import extract_features
from code.sfm.geometry import cheirality_pose, essential_from_F, reprojection_error, triangulate_dlt
from code.sfm.io_utils import save_ply, save_poses_json
from code.sfm.matching import match_images, verify_pair
from code.sfm.reconstruction import Camera, Reconstruction
from code.sfm.tracks import build_tracks


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _load_intrinsics(images: Path, names: list[str]) -> dict[str, np.ndarray]:
    candidate = images.parent / "intrinsics.yaml"
    if candidate.exists():
        payload = yaml.safe_load(candidate.read_text()).get("images", {})
        return {name: np.asarray(payload[name]["K"], dtype=float) for name in names}
    import cv2
    first = cv2.imread(str(images / names[0]))
    h, w = first.shape[:2]
    f = 1.2 * max(w, h)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.]])
    return {name: K.copy() for name in names}


def _triangulate_track(recon: Reconstruction, track: dict[int, int], keypoints: list[np.ndarray]) -> np.ndarray | None:
    registered = [image_id for image_id in track if recon.cameras[image_id].registered]
    if len(registered) < 2:
        return None
    observations = np.asarray([keypoints[i][track[i]] for i in registered])
    cameras = [recon.cameras[i] for i in registered]
    try:
        X = triangulate_dlt(observations, [camera.R for camera in cameras], [camera.t for camera in cameras], [camera.K for camera in cameras])
    except (ValueError, np.linalg.LinAlgError):
        return None
    errors = [reprojection_error(X, xy, camera.K, camera.R, camera.t) for xy, camera in zip(observations, cameras)]
    if all((camera.R @ X + camera.t)[2] > 0 for camera in cameras) and np.median(errors) < 2.0:
        return X
    return None


def _run_ba(recon: Reconstruction, keypoints: list[np.ndarray], max_nfev: int, logger: logging.Logger) -> None:
    ids = recon.registered_ids()
    if len(ids) < 2 or not recon.points:
        return
    point_ids = list(recon.points)
    point_slot = {track: slot for slot, track in enumerate(point_ids)}
    camera_slot = {image: slot for slot, image in enumerate(ids)}
    cam_idx: list[int] = []
    point_idx: list[int] = []
    observations: list[np.ndarray] = []
    Ks: list[np.ndarray] = []
    for track_id in point_ids:
        for image_id, keypoint_id in recon.tracks[track_id].items():
            if image_id in camera_slot:
                cam_idx.append(camera_slot[image_id])
                point_idx.append(point_slot[track_id])
                observations.append(keypoints[image_id][keypoint_id])
                Ks.append(recon.cameras[image_id].K)
    if len(observations) < 8:
        return
    result = bundle_adjust(
        np.asarray([recon.cameras[i].R for i in ids]), np.asarray([recon.cameras[i].t for i in ids]),
        np.asarray([recon.points[i] for i in point_ids]), np.asarray(cam_idx), np.asarray(point_idx), np.asarray(observations), np.asarray(Ks), max_nfev=max_nfev,
    )
    for slot, image_id in enumerate(ids):
        recon.cameras[image_id].R, recon.cameras[image_id].t = result.rotations[slot], result.translations[slot]
    for slot, track_id in enumerate(point_ids):
        recon.points[track_id] = result.points[slot]
    logger.info("BA RMS %.3f px -> %.3f px (%d evaluations)", result.initial_rms, result.final_rms, result.nfev)


def run_sfm(images: Path, output: Path, config: dict, max_images: int | None = None) -> Reconstruction:
    import cv2

    logger = logging.getLogger("sfm")
    names = sorted(path.name for path in images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if max_images:
        names = names[:max_images]
    if len(names) < 2:
        raise ValueError("SfM needs at least two images")
    Ks = _load_intrinsics(images, names)
    color_images = [cv2.imread(str(images / name)) for name in names]
    keypoints, descriptors = [], []
    settings = config["features"]
    for image in color_images:
        xy, desc = extract_features(image, settings["detector"], settings["n_features"], settings.get("rootsift", False))
        keypoints.append(xy); descriptors.append(desc)
    logger.info("features: median=%d", int(np.median([len(x) for x in keypoints])))
    pairs: dict[tuple[int, int], np.ndarray] = {}
    matrices: dict[tuple[int, int], np.ndarray] = {}
    binary = settings["detector"] == "orb"
    matcher_cfg = config["matching"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            tentative = match_images(descriptors[i], descriptors[j], matcher_cfg["ratio"], matcher_cfg["mutual"], binary)
            if len(tentative) < 8:
                continue
            try:
                F, inliers = verify_pair(keypoints[i], keypoints[j], tentative, threshold=matcher_cfg["ransac_thresh_px"], iterations=matcher_cfg["ransac_iters"], seed=config["seed"] + i * 1000 + j)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            if len(inliers) >= matcher_cfg["min_inliers"]:
                pairs[(i, j)], matrices[(i, j)] = inliers, F
    if not pairs:
        raise RuntimeError("no image pair met the geometric-verification threshold")
    (i0, i1), init_matches = max(pairs.items(), key=lambda entry: len(entry[1]))
    F = matrices[(i0, i1)]
    p0, p1 = keypoints[i0][init_matches[:, 0]], keypoints[i1][init_matches[:, 1]]
    R1, t1 = cheirality_pose(essential_from_F(F, Ks[names[i0]], Ks[names[i1]]), p0, p1, Ks[names[i0]], Ks[names[i1]])
    recon = Reconstruction(cameras={idx: Camera(idx, name, Ks[name], np.eye(3), np.zeros(3), False) for idx, name in enumerate(names)})
    recon.cameras[i0].registered = True
    recon.cameras[i1].R, recon.cameras[i1].t, recon.cameras[i1].registered = R1, t1, True
    recon.tracks = build_tracks(pairs)
    for track_id, track in recon.tracks.items():
        X = _triangulate_track(recon, track, keypoints)
        if X is not None:
            recon.points[track_id] = X
    logger.info("initialized (%s, %s): %d points", names[i0], names[i1], len(recon.points))
    while True:
        best: tuple[int, np.ndarray, np.ndarray] | None = None
        for image_id in range(len(names)):
            if recon.cameras[image_id].registered:
                continue
            object_points, image_points = [], []
            for track_id, X in recon.points.items():
                keypoint_id = recon.tracks[track_id].get(image_id)
                if keypoint_id is not None:
                    object_points.append(X); image_points.append(keypoints[image_id][keypoint_id])
            if best is None or len(object_points) > len(best[1]):
                best = (image_id, np.asarray(object_points), np.asarray(image_points))
        if best is None or len(best[1]) < config["registration"]["min_inliers"]:
            break
        image_id, object_points, image_points = best
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, Ks[names[image_id]], None, reprojectionError=config["registration"]["pnp_reproj_px"], iterationsCount=config["registration"]["pnp_iters"], flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < config["registration"]["min_inliers"]:
            logger.warning("PnP rejected %s", names[image_id]); break
        rvec, tvec = cv2.solvePnPRefineLM(object_points[inliers[:, 0]], image_points[inliers[:, 0]], Ks[names[image_id]], None, rvec, tvec)
        recon.cameras[image_id].R = cv2.Rodrigues(rvec)[0]
        recon.cameras[image_id].t = tvec.reshape(3)
        recon.cameras[image_id].registered = True
        added = 0
        for track_id, track in recon.tracks.items():
            if track_id not in recon.points:
                X = _triangulate_track(recon, track, keypoints)
                if X is not None:
                    recon.points[track_id] = X; added += 1
        logger.info("registered %s; added=%d, points=%d", names[image_id], added, len(recon.points))
    _run_ba(recon, keypoints, config["ba"]["max_nfev"], logger)
    colors = []
    for track_id in recon.points:
        image_id, keypoint_id = next(iter(recon.tracks[track_id].items()))
        x, y = np.round(keypoints[image_id][keypoint_id]).astype(int)
        image = color_images[image_id]
        colors.append(image[np.clip(y, 0, image.shape[0]-1), np.clip(x, 0, image.shape[1]-1)][::-1])
    output.mkdir(parents=True, exist_ok=True)
    save_poses_json(recon, output / "poses.json")
    save_ply(np.asarray(list(recon.points.values())), np.asarray(colors), output / "points.ply")
    summary = {"registered": len(recon.registered_ids()), "total_images": len(names), "points": len(recon.points)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return recon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    started = time.perf_counter()
    recon = run_sfm(args.images, args.out, _load_config(args.config), args.max_images)
    print(f"registered {len(recon.registered_ids())}/{len(recon.cameras)} images and {len(recon.points)} points in {time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
