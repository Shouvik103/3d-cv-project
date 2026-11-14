"""Create a bounded portable review/submission ZIP, never including raw datasets."""
from pathlib import Path
import argparse,zipfile,json
ROOT=Path(__file__).resolve().parents[2]


def package(out,report,demo):
    out=Path(out);report=Path(report);demo=Path(demo)
    if not report.is_file() or not demo.is_file():raise ValueError('Provide an existing PDF report and MP4 demo')
    entries=[]
    for directory in ['code','configs','tests','sample_data','docs','notebooks','validation/mesh']:
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file() or p.name.startswith('.') or '__pycache__' in p.parts:continue
            if 'cache' in p.parts and 'depth_mono' not in p.parts:continue
            entries.append((p,p.relative_to(ROOT).as_posix()))
    for name in ['README.md','requirements.txt','requirements-colmap.txt','requirements-report.txt','NOTES.md','TEAM_CONTRIBUTIONS.md','MESH_ADDON.md']:
        if (ROOT/name).is_file():entries.append((ROOT/name,name))
    entries.extend([(report,'report.pdf'),(demo,'demo.mp4')])
    sample_bytes=sum(p.stat().st_size for p,n in entries if n.startswith('sample_data/'))
    if sample_bytes>40_000_000:raise ValueError('sample_data exceeds the 40 MB budget')
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,n in sorted(entries,key=lambda e:e[1]):z.write(p,n)
        z.writestr('PACKAGE_STATUS.md','This archive contains the supplied report and demo. The generated mesh addendum and mesh segment are review assets, not the final full IEEE report, narrated course demo, or portal submission. Replace --report and --demo with final course artifacts before submission.\n')
    if out.stat().st_size>=200_000_000:out.unlink();raise ValueError('Archive exceeds 200 MB')
    result=dict(archive=out.name,compressed_bytes=out.stat().st_size,sample_bytes=sample_bytes,files=len(entries),excluded=['raw datasets','model weights','virtual environments','feature caches','large local run outputs'])
    out.with_suffix('.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--demo',type=Path,required=True);a=p.parse_args();print(json.dumps(package(a.out,a.report,a.demo),indent=2))
