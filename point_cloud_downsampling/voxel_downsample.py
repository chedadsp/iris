import numpy as np

def voxel_downsample(points, voxel_size):
    # 1. Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # 2. Find unique voxels and first point index per voxel
    _, unique_indices = np.unique(
        voxel_indices,
        axis=0,
        return_index=True
    )

    # 3. Select representative points
    return points[unique_indices]