"""Conservative progress for runs started before work-counter callbacks existed."""
from pathlib import Path
import re


def legacy_progress(folder, metadata, state):
    if state.get('status') != 'running' or state.get('stage') != 'reconstructing' or state.get('progress_units'):
        return state
    log = folder / 'run.log'
    if not log.exists():
        return state
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 65536))
        tail = stream.read().decode(errors='replace')
    names = sorted(Path(item['stored']).stem + '.jpg' for item in metadata['uploads'])
    count = len(names)
    if count < 2:
        return state
    positions = {name: i for i, name in enumerate(names)}
    candidates = []
    for match in re.finditer(r'Features ([^\n]+?): \d+', tail):
        if match[1] in positions:
            done = positions[match[1]] + 1
            candidates.append((15+7*done/count, f'Extracting features · {done}/{count} images'))
    for match in re.finditer(r'Pair ([^/\n]+)/([^:\n]+): \d+ inliers', tail):
        if match[1] in positions and match[2] in positions:
            i, j = positions[match[1]], positions[match[2]]
            done = i * (2*count-i-1) // 2 + j-i
            total = count * (count-1) // 2
            candidates.append((22+20*done/total, f'Matching views · at least {done:,}/{total:,} pairs checked'))
    for match in re.finditer(r"State: \{'registered': (\d+)", tail):
        done = int(match[1])
        candidates.append((45+7*done/count, f'Estimating camera poses · {done}/{count} registered'))
    if not candidates:
        return state
    value, message = max(candidates, key=lambda item: item[0])
    return dict(state, progress=round(max(state.get('progress', 0), value), 1), message=message)
