"""Bounded colored observations from accepted, consistency-filtered depth maps."""
import numpy as np


def depth_samples(frames,limit=1000000):
    if not frames or limit<1:raise ValueError('Depth samples need frames and a positive limit')
    budgets=np.full(len(frames),limit//len(frames),dtype=int);budgets[:limit%len(frames)]+=1
    points,colors,records=[],[],[]
    for frame,budget in zip(frames,budgets):
        depth=frame['depth'];height,width=depth.shape
        valid=np.flatnonzero(np.isfinite(depth.ravel())&(depth.ravel()>0))
        chosen=valid[np.linspace(0,len(valid)-1,min(len(valid),int(budget)),dtype=int)] if len(valid) and budget else np.empty(0,dtype=int)
        y,x=np.divmod(chosen,width);uv=np.column_stack([x,y,np.ones(len(chosen))])
        camera=(uv@np.linalg.inv(frame['K']).T)*depth[y,x,None]
        world=(camera-np.asarray(frame['t']).reshape(3))@frame['R']
        points.append(world.astype('f4'));colors.append(frame['color'][y,x].astype('u1'))
        records.append(dict(index=int(frame['index']),available_pixels=len(valid),exported_points=len(chosen)))
    return np.concatenate(points),np.concatenate(colors),dict(limit=limit,available_pixels=sum(r['available_pixels'] for r in records),exported_points=sum(r['exported_points'] for r in records),views=records,
        scope='Positive, finite depth estimates from the selected backend, after foreground and multi-view consistency filtering. Uniform per-view budgets; no interpolation or repeated sampling of a depth pixel, and no unseen-surface completion.')


def export_depth_samples(frames,path,limit=1000000):
    import open3d as o3d
    points,colors,metadata=depth_samples(frames,limit)
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(points);cloud.colors=o3d.utility.Vector3dVector(colors.astype(float)/255)
    if not o3d.io.write_point_cloud(str(path),cloud):raise OSError('Could not export depth samples')
    return metadata
