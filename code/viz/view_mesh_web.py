"""Export an embedded interactive triangle mesh: no local server or mesh fetch.

All input triangles are retained. Browser positions use 16-bit display-only
quantization; source PLY/OBJ/GLB files remain unchanged. Assets use Three.js r128.
"""
from pathlib import Path
import argparse,base64,gzip,hashlib,html,json,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mesh.mesh_export import read_mesh
from mesh.rendering import orient_for_display
from runtime import write_json

ASSETS=Path(__file__).resolve().parent/'assets'


def export_viewer(mesh_path,out,title='Interactive triangle mesh',poses=None,front_image=None,fragment_out=None,scene_id=None):
    mesh_path=Path(mesh_path);out=Path(out);mesh=read_mesh(mesh_path)
    source_hash=hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    transform=orient_for_display(mesh,json.loads(Path(poses).read_text()),front_image) if poses else np.eye(4)
    vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles)
    if len(vertices)>65535:raise ValueError('Web export supports up to 65,535 vertices; use mesh_light.ply or decimate first')
    if not np.isfinite(vertices).all() or triangles.min()<0 or triangles.max()>=len(vertices):raise ValueError('Invalid mesh positions or triangle indices')
    lo,hi=vertices.min(0),vertices.max(0);center=(lo+hi)/2;divisor=float(max(hi-lo))/2
    if divisor<=0:raise ValueError('Mesh has no spatial extent')
    vertices=(vertices-center)/divisor;lo=vertices.min(0);scale=np.ptp(vertices,axis=0)
    safe=np.where(scale>0,scale,1);quantized=np.rint((vertices-lo)/safe*65535).astype('<u2')
    colors=np.rint(np.clip(np.asarray(mesh.vertex_colors),0,1)*255).astype('u1') if mesh.has_vertex_colors() else np.full((len(vertices),3),170,'u1')
    data=quantized.tobytes()+colors.tobytes();data+=bytes(len(data)%2);index_offset=len(data);data+=triangles.astype('<u2').tobytes()
    compressed=gzip.compress(data,compresslevel=9,mtime=0)
    scene_id=scene_id or 'surface-mesh-'+source_hash[:12]
    if not re.fullmatch(r'[a-z][a-z0-9-]*',scene_id):raise ValueError('scene-id must contain lowercase ASCII letters, digits and hyphens')
    payload=dict(scene=scene_id,title=title,vertices=len(vertices),triangles=len(triangles),offset=lo.tolist(),scale=scale.tolist(),indexOffset=index_offset,byteLength=len(data),encoded=base64.b64encode(compressed).decode())
    counts=f'All {len(triangles):,} triangles · {len(vertices):,} vertices'
    fragment=(ASSETS/'mesh_viewer.html').read_text().replace('__ROOT__',scene_id).replace('__TITLE__',html.escape(title,quote=True)).replace('__COUNTS__',counts).replace('__DATA__',json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    document=(ASSETS/'mesh_standalone.html').read_text().replace('__TITLE__',html.escape(title,quote=True)).replace('__STATE_KEY__',json.dumps('codex-mesh:'+scene_id)).replace('__FRAGMENT__',fragment)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(document)
    if fragment_out:
        if len(fragment.encode())>=1_000_000:raise ValueError('Inline mesh exceeds 1 MB; use a smaller light mesh')
        fragment_out=Path(fragment_out);fragment_out.parent.mkdir(parents=True,exist_ok=True);fragment_out.write_text(fragment)
    # Verify encoding offline: no face dropped, original colors retained, tiny
    # quantization error only in display coordinates. Source data is untouched.
    restored=lo+quantized.astype(float)/65535*scale
    assert np.max(abs(restored-vertices))<=max(safe)/65535*.51
    assert np.array_equal(np.frombuffer(data,dtype='<u2',offset=index_offset).reshape(-1,3),triangles)
    assert hashlib.sha256(mesh_path.read_bytes()).hexdigest()==source_hash
    result=dict(source_mesh=mesh_path.name,source_sha256=source_hash,vertices=len(vertices),triangles=len(triangles),all_triangles_displayed=True,fragment_bytes=len(fragment.encode()),standalone_bytes=len(document.encode()),display_transform=transform.tolist(),display_center=center.tolist(),display_scale=divisor,position_quantization_bits=16,source_geometry_changed=False,renderer='Three.js 0.128.0 triangle surface',modes=['colored','shaded','wireframe','normals'])
    write_json(out.with_suffix('.json'),result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--title',default='Interactive triangle mesh');p.add_argument('--poses',type=Path);p.add_argument('--front-image');p.add_argument('--fragment-out',type=Path);p.add_argument('--scene-id');a=p.parse_args()
    print(json.dumps(export_viewer(a.mesh,a.out,a.title,a.poses,a.front_image,a.fragment_out,a.scene_id),indent=2))

if __name__=='__main__':main()
