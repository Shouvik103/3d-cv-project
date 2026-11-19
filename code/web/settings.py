"""Mesh presets shared by upload validation and reconstruction workers."""
MESH_QUALITIES = {
    'balanced': dict(label='Balanced', depth_side=640, voxel_div=192),
    'high': dict(label='High', depth_side=960, voxel_div=256),
    'ultra': dict(label='Ultra', depth_side=1280, voxel_div=384),
}
METHOD_NAMES = {'ours':'Our SfM','colmap':'COLMAP'}
MAX_UPLOAD_VIEWS = 200


def method_settings(settings):
    import math
    method=settings.get('method','ours')
    if not isinstance(method,str) or method not in (*METHOD_NAMES,'all'):
        raise ValueError('Choose Our SfM, COLMAP or Compare both')
    ratio=settings.get('matching_ratio',.85)
    focal=settings.get('focal_px')
    if focal is not None and (isinstance(focal,bool) or not isinstance(focal,(int,float)) or not 50<=focal<=100000):
        raise ValueError('Focal length must be between 50 and 100,000 pixels')
    depth=settings.get('depth_backend','sgbm')
    if depth not in ('mono','sgbm'):raise ValueError('Choose monocular depth or SGBM stereo for classical methods')
    if isinstance(ratio,bool) or not isinstance(ratio,(int,float)) or not .5<=ratio<=.95:
        raise ValueError('Our SfM matching ratio must be between 0.5 and 0.95')
    K=settings.get('camera_matrix')
    if K is not None:
        if not isinstance(K,list) or len(K)!=3 or any(not isinstance(row,list) or len(row)!=3 for row in K):
            raise ValueError('Camera matrix must be a 3 by 3 array')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for row in K for v in row):
            raise ValueError('Camera matrix must contain finite numbers')
        if not (50<=K[0][0]<=100000 and 50<=K[1][1]<=100000 and K[0][2]>=0 and K[1][2]>=0 and
                K[0][1]==0 and K[1][0]==0 and K[2]==[0,0,1]):
            raise ValueError('Use positive fx/fy, nonnegative cx/cy, zero skew and last row [0,0,1]')
        if settings.get('focal_px') is not None:raise ValueError('Supply either a full camera matrix or a focal length')
    profile=settings.get('quality_profile','object')
    if profile not in ('object','standard','quality'):raise ValueError('Choose object, standard or quality reconstruction profile')
    prior=settings.get('camera_prior')
    if prior is not None:
        if not isinstance(prior,dict) or not isinstance(prior.get('focal_ratio'),(int,float)) or isinstance(prior.get('focal_ratio'),bool) or not .2<=prior['focal_ratio']<=5:raise ValueError('Camera prior needs a focal / image-long-side ratio between 0.2 and 5')
        D=prior.get('distortion',[0,0,0,0,0])
        if not isinstance(D,list) or len(D)!=5 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>5 for v in D):raise ValueError('Invalid camera-prior distortion')
        if not isinstance(prior.get('source'),str) or not prior['source'].strip():raise ValueError('State the source of the intrinsic prior')
        if K is not None or focal is not None:raise ValueError('Use one calibration input')
        if not isinstance(prior.get('fixed',True),bool):raise ValueError('Invalid prior calibration lock')
        prior=dict(focal_ratio=float(prior['focal_ratio']),distortion=D,source=prior['source'][:300],fixed=prior.get('fixed',True))
    return dict(camera_prior=prior,method=method,matching_ratio=float(ratio),camera_matrix=K,depth_backend=depth,quality_profile=profile)


def mesh_quality(metadata):
    key = metadata.get('mesh_quality', 'balanced')
    if not isinstance(key, str) or key not in MESH_QUALITIES:
        raise ValueError('Choose Balanced, High or Ultra mesh quality')
    return MESH_QUALITIES[key]
