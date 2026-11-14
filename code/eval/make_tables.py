"""Turn logged evaluations into CSV and a compact LaTeX table."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--results', type=Path, default=Path('results')); p.add_argument('--out', type=Path, default=Path('report/tables'))
    a = p.parse_args(); rows = []
    for path in sorted(a.results.rglob('*.json')):
        data = json.loads(path.read_text())
        if 'metrics' not in data: continue
        m = data['metrics']
        rows.append(dict(run=str(path.relative_to(a.results)), method=data['method'], reference=data['reference_kind'], registration=m['registration_rate'],
                         ate=m['ate_rmse'], auc5=m['auc']['5'], auc10=m['auc']['10'], auc30=m['auc']['30']))
    if not rows: raise ValueError('No evaluation result JSON files found')
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    lines = ['\\begin{tabular}{lrrrr}', '\\hline', 'Run & Reg. & ATE & AUC@5 & AUC@30 \\\\', '\\hline']
    for r in rows:
        label = r['run'].replace('_', r'\_')
        lines.append(f"{label} & {r['registration']:.2f} & {r['ate']:.4f} & {r['auc5']:.3f} & {r['auc30']:.3f} " + r'\\')
    lines += ['\\hline', '\\end{tabular}']; (a.out/'results.tex').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__': main()
