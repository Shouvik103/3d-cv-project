# Video workflow validation

Input: `code/data/make_sample_video.py --out outputs/video-example.avi`.
This is a synthetic clip of the generated multi-depth scene: eight source poses,
uneven pauses, repeated views, blurred transitions, and a cut to an unrelated shot.
It is not a real recording. Generated video bytes stay outside Git.

```bash
python code/run_video.py --video outputs/video-example.avi --out outputs/video-example-final --focal-px 700
python -m pytest -q
```

The workflow chose source frames 0, 15, 42, 87, 126, 144, spanning 0.00 to 12.00
seconds. It reconstructed all six selected views with 2,440 points. Selection
scores, every candidate decision, original timestamps, video hash, package
versions, and source revision accompany the retained outputs. The quality
threshold rejected seven sampled blurred candidates. Six candidates from the
unrelated final shot were excluded. Fifty-seven candidates were examined.

`evaluation.json` compares reconstructed cameras to the exact original rendering
poses. `tests.txt` contains the 26-test suite result. `selected_frames.png` is the
actual exported-frame contact sheet; `preview.png` shows the resulting sparse
cloud. The extraction-only command was also checked. Real handheld video,
unknown focal lengths, and phone orientation/codec variations still need field
validation. Approximate intrinsics and geometric failures are reported explicitly.
