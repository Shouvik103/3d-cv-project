"""Retain compact measured experiment evidence without images, weights or raw arrays."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.protocol import ROOT,ORDER,sha256
from runtime import write_json


def report(root,destination):
    root=Path(root).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    ledger=json.loads((root/'experiment_status.json').read_text())
    if any(ledger['experiments'][key]['status'] not in ('completed','limited') for key in ORDER):
        raise ValueError('Finish/address every stage before publishing the consolidated measured record')
    selected=[root/'experiment_status.json',root/'protocol.yaml']
    if (root/'verification.json').exists(): selected.append(root/'verification.json')
    for stage in ORDER:
        for name in ['results.csv','results.json','RESULTS.md','comparison.png','comparison.pdf','camera_error_by_view.png','camera_error_by_view.pdf','input_conditions.png','side_by_side.png']:
            path=root/stage/name
            if path.exists():selected.append(path)
        if stage=='E1':selected+=list((root/stage/'figures').glob('*'))
        # Small per-attempt evidence only; no raw images/geometry/weights/NPZ files.
        for name in ['performance.json','evaluation.json','gt_evaluation.json','summary.json','status.json','variant.yaml','reuse.json','raw.json','REASON.md']:
            selected+=list((root/stage).rglob(name))
    if (root/'E1/environment_check.json').exists():selected.append(root/'E1/environment_check.json')
    selected+=list((root/'datasets').rglob('inputs.json'))
    for name in ['gt_provenance.json','camera_convention_check.json','source_attempts.json']:
        path=root/'M4'/name
        if path.exists():selected.append(path)
    for stage,method in [('M4','ours'),('M5','colmap'),('M5','vggt')]:
        for name in ['gt_comparison.png','gt_comparison.pdf','interactive_mesh.html','interactive_mesh.json']:
            path=root/stage/method/name
            if path.exists():selected.append(path)
    records=[]
    for source in sorted(set(selected)):
        target=destination/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        records.append(dict(path=str(target.relative_to(destination)),sha256=sha256(target),bytes=target.stat().st_size))
    write_json(destination/'evidence_manifest.json',dict(source_directory=str(root),files=records,total_bytes=sum(r['bytes'] for r in records),
        exclusions='Raw photos, GT scan, model weights, dense arrays, caches, venvs, large geometry and archived superseded environments are kept locally, not copied here.'))
    lines=['# Measured experiment results — 18 November 2026','',
        'Executed sequentially: E1 → E2 → E3 → E4 → E5 → E6 → E7 → M4 → M5. '
        'Completed experiments include failed method attempts; completion does not mean every reconstruction succeeded.',
        '', '| Stage | Work executed | Status | Results |', '| --- | --- | --- | --- |']
    descriptions=dict(E1='Our SfM / CPU COLMAP / real VGGT on car, fountain, calibrated Temple',
        E2='No BA / local / local + global BA on car and Temple',E3='Car 5/10/20/30 views; Temple 5/10/16; all three methods',
        E4='SIFT / RootSIFT / ORB on car and fountain',E5='Approximate / calibrated / shared-focal BA intrinsics on Temple',
        E6='Ratio and RANSAC threshold sensitivity on car',E7='Glossy car, sparse views, controlled blur, low-texture Dino',
        M4='Original EPFL fountain GT mesh: accuracy, completeness, F-scores, error colors',
        M5='Ours + same mono backend for COLMAP; VGGT own depth; GT-aligned side-by-side')
    for stage in ORDER:
        relative=Path(os.path.relpath(destination/stage/'RESULTS.md',ROOT/'docs')).as_posix()
        lines.append(f"| {stage} | {descriptions[stage]} | {ledger['experiments'][stage]['status']} | [Measured table]({relative}) |")
    lines+=['','## Findings','',
        '- E1 Temple: our SfM registered 16/16 cameras with 0.270% aligned camera ATE; COLMAP registered 13/16 with 0.094% ATE; VGGT predicted 16/16 with 0.710% ATE. Missing-camera penalties are reflected in pose AUC, so ATE alone is insufficient.',
        '- E2 Temple: local/global BA reduced ATE from 0.377% to 0.270%. On the car, lower reprojection error did not fix wrong global geometry or missing cameras.',
        '- E3 more images did not guarantee better results. Sparse or disconnected overlap caused outright failures and two-camera partial reconstructions. Undefined global ATE is shown as missing, while valid relative-pose metrics are retained.',
        '- E4 ORB failed car initialization but reconstructed all fountain cameras. Detector performance was scene dependent.',
        '- E5 calibrated Temple K gave 0.270% ATE, approximate K 6.970%, and shared-focal BA 8.297%. The optional focal optimizer passed a controlled synthetic recovery test but failed to recover true calibration here. The user approved Temple because real car checkerboard calibration was unavailable.',
        '- E6 car results were highly sensitive to thresholds; changing the ratio to 0.7 reduced pseudo-reference ATE to 0.935% but still registered only 10/20 cameras. This does not replace the frozen baseline or establish a universal best threshold.',
        '- E7 sparse car and blurred Temple caused classical initialization failures. VGGT returned all cameras but its pose accuracy also degraded; predicted camera count is not a guarantee of correct geometry.',
        '', '## Mesh measurements','']
    mesh=json.loads((root/'M5/results.json').read_text())
    lines+=['| Source | Fused views | Faces | Accuracy % GT diagonal ↓ | Completeness % ↓ | F@1% ↑ | Render coverage |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in mesh:
        def v(k):return '—' if row.get(k) is None else f'{row[k]:.3f}'
        lines.append(f"| {row['method']} | {row.get('fused_views')} | {row.get('faces')} | {v('accuracy_percent_diagonal')} | {v('completeness_percent_diagonal')} | {v('f01')} | {v('coverage')} |")
    lines+=['', '![GT-aligned mesh comparison](../validation/experiments_20261118/M5/side_by_side.png)',
        '', 'Interactive meshes: [ours](../validation/experiments_20261118/M4/ours/interactive_mesh.html), '
        '[COLMAP](../validation/experiments_20261118/M5/colmap/interactive_mesh.html), '
        '[VGGT](../validation/experiments_20261118/M5/vggt/interactive_mesh.html).',
        '', '## Interpretation and reproducibility','',
        'Car and E1 fountain pose comparisons use COLMAP pseudo-references, not GT. Temple/Dino have supplied GT cameras; M4/M5 use the real original fountain GT triangle model. E1 car curation previously used COLMAP poses, and the relaxed configuration was previously tuned on Temple. These are development experiments, not an unseen generalization benchmark.',
        '', 'The M4/M5 comparison uses camera-center Sim(3), GT-box crop and 200k samples per direction; it is not the official EPFL visibility-aware scorer. Classical meshing shares Depth Anything V2 Small and parameters, but source landmarks/ROI/depth alignment remain coupled with camera quality. VGGT uses a different depth model, resolution and confidence gate. Rendering checks exclude maps from fusion, not from all-view pose/network inference. Coverage accompanies masked PSNR/SSIM; meshes remain partial and may be non-watertight.',
        '', 'VGGT had lower error on its retained surface in this run, while large holes remained. Full-scan completeness includes GT regions not visible in all photos or removed by ROI filtering. Its initial meshing attempt exposed a NumPy-boolean JSON serialization bug in the no-track validation path; the bug was fixed, regression-tested and rerun with identical geometry settings. The initial failed attempt is archived.',
        '', 'All VGGT inference ran locally on Apple MPS, using pinned source/checkpoint, FP16 aggregation and FP32 heads. No CUDA timing is claimed. The original NumPy-2 attempts are archived; final E1 uses the compatible isolated NumPy-1 runtime. Timing includes available model/image/kernel caches and should not be treated as a controlled microbenchmark.',
        '', 'Protocol and commands: [EXPERIMENTS.md](EXPERIMENTS.md). Compact evidence is retained in `validation/experiments_20261118/`; full input hashes, logs, photos, arrays and meshes remain in ignored `results/experiments_20261118/` and `data/`. Original GT/weights are not redistributed. No final IEEE report, narrated course demo, submission upload, or GitHub push is implied by this experiment record.',
        '', 'The final test result is recorded in `validation/experiments_20261118/verification.json`.']
    text='\n'.join(lines)+'\n';(ROOT/'docs/EXPERIMENT_RESULTS.md').write_text(text)
    relative=Path(os.path.relpath(ROOT/'docs/EXPERIMENT_RESULTS.md',root)).as_posix()
    (root/'RESULTS.md').write_text(f'# Sequential experiment results\n\n[Consolidated measured report]({relative})\n\nPer-stage tables, logs and geometry are in this directory; compact versioned evidence is in validation/experiments_20261118.\n')
    print(f'Retained {len(records)} files, {sum(r["bytes"] for r in records)/1e6:.2f} MB',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'results/experiments_20261118')
    p.add_argument('--evidence',type=Path,default=ROOT/'validation/experiments_20261118');a=p.parse_args();report(a.out,a.evidence)
