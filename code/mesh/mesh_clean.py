"""Deterministic ROI cropping, small-component removal and light mesh creation."""
from copy import deepcopy
import numpy as np


def clean(mesh,roi_box,min_component_frac=.02,light_target=80000):
    import open3d as o3d
    full=deepcopy(mesh).crop(o3d.geometry.AxisAlignedBoundingBox(*[np.asarray(x,float) for x in roi_box]))
    full.remove_degenerate_triangles();full.remove_duplicated_triangles();full.remove_duplicated_vertices();full.remove_non_manifold_edges()
    if not len(full.triangles):raise ValueError('No triangles after ROI cleaning; inspect alignment, ROI and depth coverage')
    labels,counts,_=full.cluster_connected_triangles();labels=np.asarray(labels);counts=np.asarray(counts)
    full.remove_triangles_by_mask(counts[labels]<counts.max()*min_component_frac);full.remove_unreferenced_vertices()
    light=full.simplify_quadric_decimation(int(light_target)) if len(full.triangles)>light_target else deepcopy(full)
    light.remove_degenerate_triangles();light.remove_unreferenced_vertices()
    for m in [full,light]:
        m.compute_vertex_normals()
        if m.has_vertex_colors():m.vertex_colors=o3d.utility.Vector3dVector(np.clip(np.asarray(m.vertex_colors),0,1))
    return full,light
