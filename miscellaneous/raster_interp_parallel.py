from glob import glob
import os
import re
from tqdm import tqdm
import xarray as xr
from joblib import Parallel, delayed

# Configuration
input_dir = "D:/Plate Motion Models/Muller2022/Dietmar/Phanerozoic_Muller22_grids/ErosionDeposition_present-day-coords"
output_dir = "D:/Plate Motion Models/Muller2022/Dietmar/Phanerozoic_Muller22_grids/ErosionDeposition_present-day-coords_interp"
os.makedirs(output_dir, exist_ok=True)
output_resolution = 1  # in million years

# Helper to extract age from filename
def extract_age(filename):
    match = re.search(r"_(\d+)Ma\.nc$", filename)
    return int(match.group(1)) if match else None

# Load all files and map to their ages
nc_files = glob(os.path.join(input_dir, "*.nc"))
file_age_map = {extract_age(f): f for f in nc_files if extract_age(f) is not None}
sorted_ages = sorted(file_age_map.keys())

# Interpolate to desired temporal resolution
max_age = sorted_ages[-1]
# If extrapolation required
# all_ages = list(range(sorted_ages[0], max_age + output_resolution * 2, output_resolution))
all_ages = list(range(sorted_ages[0], max_age + output_resolution, output_resolution))

def process_age(age):
    if age in file_age_map:
        # Exact match, just copy or load and save
        ds = xr.open_dataset(file_age_map[age])
        ds.to_netcdf(os.path.join(output_dir, f"erodep_{age}Ma.nc"))
    else:
        # Interpolate between surrounding known ages
        lower_ages = [a for a in sorted_ages if a < age]
        upper_ages = [a for a in sorted_ages if a > age]

        # # Extrapolation for one step beyond max age
        # if age > max(sorted_ages) and len(sorted_ages) >= 2:
        #     lower, upper = sorted_ages[-2], sorted_ages[-1]
        # elif not lower_ages or not upper_ages:
        #     continue  # skip extrapolation outside known range
        # else:
        #     lower = max(lower_ages)
        #     upper = min(upper_ages)

        lower = max(lower_ages)
        upper = min(upper_ages)

        # Load datasets
        ds_lower = xr.open_dataset(file_age_map[lower])
        ds_upper = xr.open_dataset(file_age_map[upper])

        # Weighted interpolation for variable "z"
        da_lower = ds_lower["z"]
        da_upper = ds_upper["z"]

        # Ensure alignment (same coordinates)
        da_lower, da_upper = xr.align(da_lower, da_upper)

        # Compute interpolated variable
        weight_upper = (age - lower) / (upper - lower)
        weight_lower = 1 - weight_upper
        interpolated_z = da_lower * weight_lower + da_upper * weight_upper

        # Preserve attributes
        interpolated_z.attrs = da_lower.attrs

        # Build new dataset using only "z"
        interpolated_ds = xr.Dataset(
            {"z": interpolated_z},
            coords={coord: ds_lower.coords[coord] for coord in ds_lower.coords}
        )
        interpolated_ds.attrs = ds_lower.attrs

        # Preserve attributes and coordinates
        interpolated_ds.attrs = ds_lower.attrs
        for var in interpolated_ds.data_vars:
            interpolated_ds[var].attrs = ds_lower[var].attrs

        # Save to NetCDF
        output_filename = os.path.join(output_dir, f"erodep_{age}Ma.nc")

        # Prepare compression settings for all variables
        encoding = {
            var: {
                "zlib": True,
                "complevel": 1,  # Compression level (1–9), 4 is a good balance
                "shuffle": True
            }
            for var in interpolated_ds.data_vars
        }

        interpolated_ds.to_netcdf(output_filename, encoding=encoding)

# Parallel execution with progress bar
results = Parallel(n_jobs=12)(
    delayed(process_age)(age) for age in tqdm(all_ages)
)
