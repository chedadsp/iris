import numpy as np

def voxel_downsample(scan: dict, voxel_size: float) -> dict:
    """
    Voxel downsample an E57 scan while preserving ALL point attributes.

    Parameters
    ----------
    scan : dict
        Dictionary returned by pye57.read_scan(...), e.g.
        {
            'cartesianX': np.ndarray,
            'cartesianY': np.ndarray,
            'cartesianZ': np.ndarray,
            'azimuth': np.ndarray,
            'elevation': np.ndarray,
            'range': np.ndarray,
            ...
        }

    voxel_size : float
        Voxel size in meters.

    Returns
    -------
    dict
        Downsampled scan with the same keys as input.
    """

    # 1. Stack XYZ
    xyz = np.vstack((
        scan["cartesianX"],
        scan["cartesianY"],
        scan["cartesianZ"]
    )).T  # shape (N, 3)

    # 2. Compute voxel indices
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)

    # 3. Unique voxels → representative point indices
    _, unique_indices = np.unique(
        voxel_indices,
        axis=0,
        return_index=True
    )

    # 4. Downsample ALL attributes using same indices
    downsampled_scan = {}
    for key, values in scan.items():
        if isinstance(values, np.ndarray) and len(values) == xyz.shape[0]:
            downsampled_scan[key] = values[unique_indices]
        else:
            # keep non per-point metadata untouched (if any)
            downsampled_scan[key] = values

    return downsampled_scan

def compute_spherical_from_xyz(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.mod(np.arctan2(y, x), 2 * np.pi)
    el = -(np.arcsin(z / r))
    return az, el, r