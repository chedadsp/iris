from voxel_downsample import voxel_downsample
from pye57 import E57
 
input_e57 = r"C:\MASTER_TEA\iris\data\input\scan001.e57"
output_e57 = r"C:\MASTER_TEA\iris\point_cloud_processing\data\downsampled\scan001_downsampled_2cm.e57"
 
# Open file
e57 = E57(input_e57)
 
data = e57.read_scan(0, ignore_missing_fields=True)
 
print("Number of points:", len(data["cartesianX"]))
 
ds_scan = voxel_downsample(data, voxel_size=0.02)
print("********************Downsampled keys:", ds_scan.keys()) 
print("Downsampled points:", len(ds_scan["cartesianX"]))
 
e57_out = E57(output_e57, mode="w")
e57_out.write_scan_raw(ds_scan)

e57_out.close()
e57.close()
 
e57_test = E57(output_e57)
scan_test = e57_test.read_scan(0, ignore_missing_fields=True)
print("*******************Reloaded keys:", scan_test.keys())

print("Saved:", output_e57)