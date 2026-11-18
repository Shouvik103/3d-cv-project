"""Mesh presets shared by upload validation and reconstruction workers."""
MESH_QUALITIES = {
    'balanced': dict(label='Balanced', depth_side=640, voxel_div=192),
    'high': dict(label='High', depth_side=960, voxel_div=256),
    'ultra': dict(label='Ultra', depth_side=1280, voxel_div=384),
}


def mesh_quality(metadata):
    key = metadata.get('mesh_quality', 'balanced')
    if not isinstance(key, str) or key not in MESH_QUALITIES:
        raise ValueError('Choose Balanced, High or Ultra mesh quality')
    return MESH_QUALITIES[key]
