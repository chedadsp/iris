import numpy as np
from voxel_downsample import voxel_downsample
from pye57 import E57

input_e57 = r"C:\MASTER_RAD\iris\data\lidar_croped_promenada\kola005_013.e57"
output_e57 = r"C:\MASTER_RAD\iris\data\lidar_preprocessing\kola005_013_downsampled_5cm.e57"

# Open file
e57 = E57(input_e57)

data = e57.read_scan(0, ignore_missing_fields=True)

print("Number of points:", len(data["cartesianX"]))

points = np.vstack((
    data["cartesianX"],
    data["cartesianY"],
    data["cartesianZ"]
)).T

print("Original points:", points.shape[0])

voxel_size = 0.05  # 5 cm

downsampled_points = voxel_downsample(points, voxel_size)

print("Downsampled points:", downsampled_points.shape[0])

e57_out = E57(output_e57, mode="w")

e57_out.write_scan_raw({
    "cartesianX": downsampled_points[:, 0],
    "cartesianY": downsampled_points[:, 1],
    "cartesianZ": downsampled_points[:, 2],
})

e57_out.close()
e57.close()

print("Saved:", output_e57)