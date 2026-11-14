"""Color-preserving exports. OBJ uses its de-facto per-vertex RGB extension."""
from pathlib import Path
import numpy as np


def as_trimesh(mesh):
    import trimesh
    colors=np.rint(np.clip(np.asarray(mesh.vertex_colors),0,1)*255).astype('u1') if mesh.has_vertex_colors() else None
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices),faces=np.asarray(mesh.triangles),vertex_colors=colors,process=False)


def read_mesh(path):
    import open3d as o3d, trimesh
    path=Path(path)
    if path.suffix.lower()=='.ply':mesh=o3d.io.read_triangle_mesh(str(path))
    else:
        tm=trimesh.load(path,force='mesh',process=False)
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices),o3d.utility.Vector3iVector(tm.faces))
        if tm.visual.kind=='vertex':mesh.vertex_colors=o3d.utility.Vector3dVector(np.asarray(tm.visual.vertex_colors)[:,:3]/255.)
    if not len(mesh.triangles):raise ValueError(f'{path} has no triangles; supply a triangle mesh, not a point cloud')
    mesh.compute_vertex_normals();return mesh


def export_meshes(full,light,out,formats=('ply','obj','glb')):
    import open3d as o3d
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,m in [('mesh_full',full),('mesh_light',light)]:
        if not o3d.io.write_triangle_mesh(str(out/(name+'.ply')),m,write_vertex_colors=True):raise IOError('PLY mesh export failed')
    tm=as_trimesh(light)
    for ext in formats:
        if ext=='ply':continue
        if ext not in ('obj','glb'):raise ValueError(f'Unsupported mesh export {ext}')
        data=tm.export(file_type=ext)
        p=out/('mesh_light.'+ext)
        p.write_bytes(data) if isinstance(data,bytes) else p.write_text(data)
