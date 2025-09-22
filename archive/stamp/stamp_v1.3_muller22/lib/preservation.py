'''
Preservation

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 05/07/2025
'''

import os
from sys import stderr
from typing import (
    Optional,
    Union,
)

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import (
    ArrayLike,
    NDArray,
)
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import xarray as xr


_PathLike = Union[os.PathLike, str]
_PathOrDataFrame = Union[_PathLike, pd.DataFrame]


def run_coregister_erosion_deposition(
    point_data: _PathOrDataFrame,
    input_dir: _PathLike,
    distance_threshold: float = 0.1,
    output_filename: Optional[_PathLike] = None,
    n_jobs: int = -2,
    verbose: bool = False,
) -> pd.DataFrame:

    if isinstance(point_data, str):
        point_data = pd.read_csv(point_data)
    else:
        point_data = pd.DataFrame(point_data)
    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        out = parallel(
            delayed(_coregister_erosion_deposition)(
                time=t,
                input_dir=input_dir,
                df=d,
                distance_threshold=distance_threshold,
            )
            for t, d in point_data.groupby("age (Ma)")
        )

    out = pd.DataFrame(pd.concat(out, ignore_index=True))
    if "label" in out.columns:
        sort_by = ["label", "age (Ma)"]
    else:
        sort_by = "age (Ma)"
    out = out.sort_values(by=sort_by, ignore_index=True)
    
    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.exists(output_dir):
            if verbose:
                print(
                    "Output directory does not exist; creating now: "
                    + output_dir,
                    file=stderr,
                )
            os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(
                "Writing output to file: "
                + os.path.basename(output_filename),
                file=stderr,
            )
        out.to_csv(output_filename, index=False)

    return out


def _coregister_erosion_deposition(
    time: float,
    input_dir: _PathLike,
    df: _PathOrDataFrame,
    distance_threshold: float = 0.1,
) -> pd.DataFrame:
    
    df = df.copy()
    df = df[df["age (Ma)"] == time]
    input_filename = os.path.join(
        input_dir, "cumulative_erosion_deposition_{:0.0f}Ma.nc".format(time)
    )
    with xr.open_dataset(input_filename) as dset:
        thickness = np.array(dset["z"])
        try:
            grid_lons = np.array(dset["lon"])
        except KeyError:
            grid_lons = np.array(dset["x"])
        try:
            grid_lats = np.array(dset["lat"])
        except KeyError:
            grid_lats = np.array(dset["y"])
    mlons, mlats = np.meshgrid(grid_lons, grid_lats)
    mlons = np.deg2rad(mlons[~np.isnan(thickness)])
    mlats = np.deg2rad(mlats[~np.isnan(thickness)])
    thickness = thickness[~np.isnan(thickness)]
    mcoords = np.hstack(
        (
            mlats.reshape((-1, 1)),
            mlons.reshape((-1, 1)),
        )
    )
    neigh = NearestNeighbors(metric="haversine")
    neigh.fit(mcoords)
    point_lons = np.deg2rad(np.array(df["present_lon"]))
    point_lats = np.deg2rad(np.array(df["present_lat"]))
    point_coords = np.hstack(
        (
            point_lats.reshape((-1, 1)),
            point_lons.reshape((-1, 1)),
        )
    )
    
    # Get points within radius
    distances, radius_indices = neigh.radius_neighbors(
        point_coords,
        radius=np.deg2rad(distance_threshold),
        return_distance=True,
        sort_results=True,
    )
    
    # Get nearest single point for fallback
    nearest_distances, nearest_indices = neigh.kneighbors(
        point_coords, 
        n_neighbors=1,
        return_distance=True
    )
    
    crustal_thickness = np.full(df.shape[0], np.nan)
    
    for i in range(df.shape[0]):
        indices_point = radius_indices[i]
        
        # If no points within radius, use the nearest point
        if indices_point.size == 0:
            nearest_idx = nearest_indices[i][0]
            data = np.array([thickness[nearest_idx]])
        else:
            data = thickness[indices_point]
            
        # Calculate mean thickness
        crustal_thickness[i] = np.nanmean(data)
    
    # Add the single column with the new name
    df["erosion-deposition (m)"] = crustal_thickness
    
    return df


def clean_outliers(
    data: ArrayLike,
    contamination: Union[float, str] = "auto",
    n_jobs: int = 1,
    **kwargs
) -> NDArray:
    
    data = np.array(data)
    forest = IsolationForest(
        n_jobs=n_jobs,
        contamination=contamination,
        **kwargs
    )
    if data.ndim == 1:
        x = np.reshape(data, (-1, 1))
    else:
        x = data
    forest_result = forest.fit_predict(x)
    if data.ndim == 1:
        data = data[forest_result == 1]
    else:
        data = data[forest_result == 1, :]
    return data
