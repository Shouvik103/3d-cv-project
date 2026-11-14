"""Content-based video keyframes, with bounded disk storage and exact provenance.

The first pass scores a bounded set of candidates. Motion and forward/backward
optical-flow consistency choose views within one continuous shot. The second
pass decodes the selected original frames sequentially, avoiding seek errors.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

from runtime import manifest, write_json

LOGGER = logging.getLogger('video')


@dataclass
class Candidate:
    frame_index: int
    timestamp_seconds: float
    timestamp_source: str
    sharpness: float
    corners: int
    gray_path: Path

    def record(self):
        return {k: v for k, v in asdict(self).items() if k != 'gray_path'}


def validate_settings(cfg):
    for name in ('candidate_hz', 'max_candidates', 'analysis_max_side', 'max_frames',
                 'min_frames', 'max_side', 'min_corners'):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f'{name} must be positive')
    for name in ('max_candidates', 'analysis_max_side', 'max_frames', 'min_frames', 'max_side', 'min_corners'):
        if not isinstance(cfg[name], int) or isinstance(cfg[name], bool):
            raise ValueError(f'{name} must be an integer')
    if not 2 <= cfg['min_frames'] <= cfg['max_frames']:
        raise ValueError('Require 2 <= min_frames <= max_frames')
    if cfg['max_candidates'] < cfg['max_frames']:
        raise ValueError('max_candidates must be at least max_frames')
    if not 0 < cfg['min_motion_fraction'] <= cfg['target_motion_fraction'] < cfg['max_motion_fraction'] < 1:
        raise ValueError('Motion fractions must satisfy 0 < min <= target < max < 1')
    if not 0 < cfg['min_tracked_fraction'] <= 1:
        raise ValueError('min_tracked_fraction must be in (0, 1]')
    if not 0 <= cfg['relative_sharpness'] <= 1 or not np.isfinite(cfg['min_sharpness']) or cfg['min_sharpness'] < 0:
        raise ValueError('Invalid sharpness settings')


def open_video(path):
    if not path.is_file():
        raise ValueError(f'Video file does not exist: {path}')
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError('Cannot open this video; use a locally decodable MP4, MOV, or AVI file')
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return capture


def resize_max(image, max_side):
    h, w = image.shape[:2]
    scale = min(1., max_side/max(h, w))
    size = (max(1, round(w*scale)), max(1, round(h*scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA) if size != (w, h) else image


def corners(gray):
    return cv2.goodFeaturesToTrack(gray, maxCorners=350, qualityLevel=.01, minDistance=7, blockSize=7)


def scan_video(path, work, cfg):
    cap = open_video(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no usable frame-rate metadata; re-encode it before extraction')
        reported = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        reported = int(reported) if np.isfinite(reported) and reported > 0 else 0
        stride = max(1, math.ceil(fps/cfg['candidate_hz']), math.ceil(reported/cfg['max_candidates']))
        candidates = []
        decoded_size = None
        index = 0
        last_timestamp = -1.
        while cap.grab():
            if index % stride == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise ValueError(f'Could not decode frame {index}')
                size = frame.shape[1::-1]
                if decoded_size is not None and size != decoded_size:
                    raise ValueError('Video dimensions change during the shot; split it into separate clips')
                decoded_size = size
                seconds = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                source = 'decoder_pts'
                if not np.isfinite(seconds) or seconds < 0 or seconds <= last_timestamp:
                    seconds = index/fps
                    source = 'frame_index_over_fps'
                if seconds <= last_timestamp:
                    raise ValueError('Video timestamps are inconsistent; re-encode it before extraction')
                last_timestamp = seconds
                gray = cv2.cvtColor(resize_max(frame, cfg['analysis_max_side']), cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                features = corners(gray)
                gray_path = work/f'{index:09d}.png'
                if not cv2.imwrite(str(gray_path), gray):
                    raise OSError('Could not save frame-selection working data')
                candidates.append(Candidate(index, seconds, source, sharpness,
                                            0 if features is None else len(features), gray_path))
                if len(candidates) > cfg['max_candidates']:
                    raise ValueError('Frame count metadata underestimates clip length; re-encode or increase max_candidates')
                if len(candidates) % 50 == 0:
                    LOGGER.info('Scanned %d candidate frames, at %.1f seconds', len(candidates), seconds)
            index += 1
        if len(candidates) < cfg['min_frames']:
            raise ValueError('Video is too short for the configured number of reconstruction views')
        if reported > 0 and index < reported*.95:
            raise ValueError(f'Video decoding stopped early ({index}/{reported} frames); check for corruption')
        return candidates, dict(fps=float(fps), reported_frames=reported, decoded_frames=index,
                                input_size=list(decoded_size), candidate_stride_frames=stride)
    finally:
        cap.release()


class Motion:
    """Tracked fraction is an overlap proxy, not a measured 3D parallax angle."""
    @staticmethod
    @lru_cache(maxsize=4)
    def gray(path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('Missing frame-selection working image')
        return image

    def __call__(self, a, b):
        first, second = self.gray(a.gray_path), self.gray(b.gray_path)
        p = corners(first)
        if p is None or len(p) < 8:
            return dict(tracked_fraction=0., motion_fraction=0.)
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        q, forward, error = cv2.calcOpticalFlowPyrLK(first, second, p, None, **options)
        if q is None:
            return dict(tracked_fraction=0., motion_fraction=0.)
        back, backward, _ = cv2.calcOpticalFlowPyrLK(second, first, q, None, **options)
        if back is None:
            return dict(tracked_fraction=0., motion_fraction=0.)
        xy = q.reshape(-1, 2)
        valid = (forward.ravel() == 1) & (backward.ravel() == 1)
        valid &= np.linalg.norm(back.reshape(-1, 2)-p.reshape(-1, 2), axis=1) < 1.5
        valid &= error.ravel() < 25
        valid &= (xy[:, 0] >= 0) & (xy[:, 0] < second.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < second.shape[0])
        motion = np.linalg.norm(xy[valid]-p.reshape(-1, 2)[valid], axis=1)
        return dict(tracked_fraction=float(valid.mean()),
                    motion_fraction=float(np.median(motion)/np.hypot(*first.shape)) if len(motion) else 0.)


def connected(stats, cfg):
    return stats['tracked_fraction'] >= cfg['min_tracked_fraction'] and stats['motion_fraction'] <= cfg['max_motion_fraction']


def select_frames(candidates, cfg):
    threshold = max(cfg['min_sharpness'], float(np.median([c.sharpness for c in candidates]))*cfg['relative_sharpness'])
    good = [c for c in candidates if c.sharpness >= threshold and c.corners >= cfg['min_corners']]
    if len(good) < cfg['min_frames']:
        raise ValueError('Too few sharp, textured frames; record in better light and keep the object in focus')
    measure = Motion()
    try:
        segments = [[good[0]]]
        scores = [0.]
        for previous, current in zip(good, good[1:]):
            stats = measure(previous, current)
            if not connected(stats, cfg):
                segments.append([])
                scores.append(0.)
            else:
                scores[-1] += stats['motion_fraction']
            segments[-1].append(current)
        viable = [i for i, segment in enumerate(segments) if len(segment) >= cfg['min_frames']]
        if not viable:
            raise ValueError('No continuous shot has enough overlapping views; move more slowly and avoid cuts')
        best = max(viable, key=lambda i: scores[i])
        segment = segments[best]

        def choose(target):
            selected = [segment[0]]
            for current in segment[1:]:
                stats = measure(selected[-1], current)
                if connected(stats, cfg) and stats['motion_fraction'] >= target:
                    selected.append(current)
            return selected

        selected = choose(cfg['target_motion_fraction'])
        target = cfg['target_motion_fraction']
        # A short but useful clip can need denser views than the nominal target.
        while len(selected) < cfg['min_frames'] and target > cfg['min_motion_fraction']:
            target = max(cfg['min_motion_fraction'], target/2)
            selected = choose(target)
        if len(selected) < cfg['min_frames']:
            raise ValueError('Too little viewpoint change; move the camera around the stationary object')
        if len(selected) > cfg['max_frames']:
            indices = np.linspace(0, len(selected)-1, cfg['max_frames'], dtype=int)
            selected = [selected[i] for i in indices]
        links = [measure(a, b) for a, b in zip(selected, selected[1:])]
        if not all(connected(link, cfg) for link in links):
            raise ValueError('Cannot preserve overlap within max_frames; shorten the clip or increase --max-frames')
        # Do not silently stop at an unrecoverable tracking gap inside the selected shot.
        tail = measure(selected[-1], segment[-1]) if selected[-1] != segment[-1] else None
        if tail and not connected(tail, cfg):
            raise ValueError('Tracking lost before the end of the shot; use a slower, steadier camera path')
        return selected, dict(sharpness_threshold=threshold, motion_threshold_fraction=target,
                              continuous_shots=len(segments), selected_shot=best,
                              shot_start_seconds=segment[0].timestamp_seconds,
                              shot_end_seconds=segment[-1].timestamp_seconds, links=links)
    finally:
        measure.gray.cache_clear()


def save_frames(video, selected, images, cfg, calibration=None, focal_px=None):
    calibration_data = yaml.safe_load(calibration.read_text()) if calibration else None
    if calibration and focal_px is not None:
        raise ValueError('Choose calibration or focal length, not both')
    if focal_px is not None and (not np.isfinite(focal_px) or focal_px <= 0):
        raise ValueError('Focal length must be finite and positive')
    wanted = {c.frame_index: c for c in selected}
    cap = open_video(video)
    records, exported = {}, []
    index = 0
    images.mkdir(parents=True, exist_ok=True)
    try:
        while wanted and cap.grab():
            if index in wanted:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise ValueError(f'Cannot decode selected source frame {index}')
                h, w = frame.shape[:2]
                if calibration_data:
                    if calibration_data['image_size'] != [w, h]:
                        raise ValueError('Video dimensions do not match checkerboard calibration')
                    original = np.array(calibration_data['K'], float)
                    distortion = np.array(calibration_data['distortion'], float)
                    K, _ = cv2.getOptimalNewCameraMatrix(original, distortion, (w, h), 0, (w, h))
                    frame = cv2.undistort(frame, original, distortion, None, K)
                    source = 'checkerboard_undistorted'
                else:
                    f = focal_px if focal_px is not None else 1.2*max(w, h)
                    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1.]])
                    source = 'supplied_video_focal_px' if focal_px is not None else 'fallback_1.2_max_side'
                frame = resize_max(frame, cfg['max_side'])
                K[0] *= frame.shape[1]/w
                K[1] *= frame.shape[0]/h
                name = f'frame_{index:09d}.jpg'
                if not cv2.imwrite(str(images/name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f'Cannot save {name}')
                records[name] = dict(K=K.tolist(), image_size=list(frame.shape[1::-1]), source=source)
                exported.append(dict(wanted.pop(index).record(), image_name=name))
            index += 1
        if wanted:
            raise ValueError('Video ended before the selected frames could be exported')
    finally:
        cap.release()
    (images.parent/'intrinsics.yaml').write_text(yaml.safe_dump(dict(images=records)))
    return exported


def contact_sheet(images, records, path):
    width, height, columns = 240, 200, 4
    sheet = Image.new('RGB', (columns*width, math.ceil(len(records)/columns)*height), '#171c24')
    draw = ImageDraw.Draw(sheet)
    for i, record in enumerate(records):
        with Image.open(images/record['image_name']) as image:
            image.thumbnail((width, height-25))
            x, y = (i % columns)*width, (i//columns)*height
            sheet.paste(image, (x+(width-image.width)//2, y))
            draw.text((x+6, y+height-21), f"{record['timestamp_seconds']:.2f}s  frame {record['frame_index']}", fill='white')
    sheet.save(path)


def extract_video(video, output, cfg, calibration=None, focal_px=None):
    validate_settings(cfg)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Frame output directory must be empty; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    with TemporaryDirectory(prefix='frame-scan-', dir=output) as scratch:
        candidates, metadata = scan_video(video, Path(scratch), cfg)
        selected, selection = select_frames(candidates, cfg)
        records = save_frames(video, selected, output/'images', cfg, calibration, focal_px)
    digest = hashlib.sha256()
    with video.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    chosen = {r['frame_index'] for r in records}
    decisions = []
    for candidate in candidates:
        if candidate.frame_index in chosen:
            reason = 'selected'
        elif candidate.sharpness < selection['sharpness_threshold']:
            reason = 'blur'
        elif candidate.corners < cfg['min_corners']:
            reason = 'low_texture'
        elif not selection['shot_start_seconds'] <= candidate.timestamp_seconds <= selection['shot_end_seconds']:
            reason = 'other_shot'
        else:
            reason = 'not_needed_by_motion_overlap_or_frame_budget'
        decisions.append(dict(candidate.record(), selected=candidate.frame_index in chosen, decision=reason))
    selection['candidates'] = decisions
    report = dict(manifest(cfg), video_name=video.name, video_sha256=digest.hexdigest(), video=metadata,
                  selected_frames=records, selection=selection,
                  intrinsics_source='checkerboard' if calibration else 'supplied_focal_px' if focal_px else 'approximate',
                  selection_scope='Most motion-rich continuous shot with enough sharp, textured candidates')
    write_json(output/'frames.json', report)
    contact_sheet(output/'images', records, output/'selected_frames.png')
    LOGGER.info('Selected %d/%d candidate frames, from %.2fs to %.2fs', len(records), len(candidates),
                records[0]['timestamp_seconds'], records[-1]['timestamp_seconds'])
    return report
