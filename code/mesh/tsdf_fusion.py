"""Open3D scalable TSDF with OpenCV world-to-camera extrinsics."""
import numpy as np


def fuse(frames,voxel_length,sdf_trunc,depth_trunc):
    import open3d as o3d
    if voxel_length<=0 or sdf_trunc<=0:raise ValueError('TSDF scales must be positive')
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=float(voxel_length),sdf_trunc=float(sdf_trunc),color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for f in frames:
        d=np.ascontiguousarray(f['depth'],dtype='f4');color=np.ascontiguousarray(f['color'],dtype='u1');h,w=d.shape;K=f['K']
        if color.shape[:2]!=(h,w):raise ValueError('Frame color and depth must have equal resolution; scale K before fusion')
        extrinsic=np.eye(4);extrinsic[:3,:3]=f['R'];extrinsic[:3,3]=f['t']
        intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2],K[1,2])
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color),o3d.geometry.Image(d),depth_scale=1.,depth_trunc=float(depth_trunc),convert_rgb_to_intensity=False)
        volume.integrate(rgbd,intrinsic,extrinsic)
    mesh=volume.extract_triangle_mesh();mesh.compute_vertex_normals()
    return mesh,volume
