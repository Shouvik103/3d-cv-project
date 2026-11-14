"""Optional oriented dense-point Poisson ablation, not the default surface."""
import numpy as np


def reconstruct(points,depth=8,density_quantile=.05):
    import open3d as o3d
    if len(points.points)<10:raise ValueError('Poisson requires a nonempty dense cloud')
    if not points.has_normals():
        points.estimate_normals();points.orient_normals_consistent_tangent_plane(min(30,len(points.points)-1))
    mesh,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(points,depth=int(depth),n_threads=4)
    mesh.remove_vertices_by_mask(np.asarray(density)<np.quantile(density,density_quantile));mesh.compute_vertex_normals();return mesh
