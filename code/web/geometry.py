"""Prepare bounded browser geometry without changing exported reconstruction files."""
from pathlib import Path
import json

import numpy as np


def render_preview(source,poses,output):
    from mesh.mesh_export import read_mesh
    from mesh.rendering import Renderer,orbit_camera,orient_for_display
    from PIL import Image
    mesh=read_mesh(source)
    try:orient_for_display(mesh,json.loads(Path(poses).read_text()))
    except ValueError:pass
    K,R,t=orbit_camera(mesh,angle=.45,elevation=.25,width=480,height=320)
    Image.fromarray(Renderer(mesh).render(K,R,t,480,320)['color']).save(output)


def export_geometry(source, output, poses=None, kind='cloud', limit=115000):
    import open3d as o3d
    source, output = Path(source), Path(output)
    if kind == 'mesh':
        obj = o3d.io.read_triangle_mesh(str(source))
        points = np.asarray(obj.vertices).copy()
        colors = np.asarray(obj.vertex_colors).copy()
        triangles = np.asarray(obj.triangles).copy()
        if not len(triangles):
            raise ValueError('No mesh triangles were produced')
    else:
        obj = o3d.io.read_point_cloud(str(source))
        points, colors = np.asarray(obj.points).copy(), np.asarray(obj.colors).copy()
        triangles = None
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('The output contains no usable geometry')
    total = len(points)
    # Camera-derived upright axes match the existing mesh interface. The same
    # rigid transform works on points; original files retain their native frame.
    transform = np.eye(4)
    if poses and Path(poses).exists():
        from mesh.rendering import orient_for_display
        proxy = o3d.geometry.TriangleMesh()
        proxy.vertices = o3d.utility.Vector3dVector(points)
        try:
            transform = orient_for_display(proxy, json.loads(Path(poses).read_text()))
            points = np.asarray(proxy.vertices).copy()
        except ValueError:
            pass
    if len(colors) != total:
        colors = np.full((total, 3), .65)
    if triangles is None and total > limit:
        keep = np.sort(np.random.default_rng(0).choice(total, limit, replace=False))
        points, colors = points[keep], colors[keep]
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    divisor = float(np.max(hi - lo)) / 2 or 1
    payload = dict(kind=kind, total_points=total, displayed_points=len(points),
                   triangles=len(triangles) if triangles is not None else 0,
                   positions=np.round((points - center) / divisor, 6).ravel().tolist(),
                   colors=np.rint(np.clip(colors, 0, 1) * 255).astype('u1').ravel().tolist(),
                   indices=triangles.ravel().tolist() if triangles is not None else None,
                   coordinate_frame='Normalized upright display coordinates; arbitrary units',
                   display_transform=transform.tolist(), display_center=center.tolist(), display_scale=divisor)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    return payload
