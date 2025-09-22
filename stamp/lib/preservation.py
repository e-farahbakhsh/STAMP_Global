'''
Preservation

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 16/09/2025
'''

import os
from sys import stderr
from typing import (
    Optional,
    Tuple,
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


def run_coregister_erosion(
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
            delayed(_coregister_erosion)(
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


def _coregister_erosion(
    time: float,
    input_dir: _PathLike,
    df: _PathOrDataFrame,
    distance_threshold: float = 0.1,
) -> pd.DataFrame:
    
    df = df.copy()
    df = df[df["age (Ma)"] == time]
    input_filename = os.path.join(
        input_dir, "cumulative_erosion_{:0.0f}Ma.nc".format(time)
    )
    with xr.open_dataset(input_filename) as dset:
        erosion = np.array(dset["z"])
        try:
            grid_lons = np.array(dset["lon"])
        except KeyError:
            grid_lons = np.array(dset["x"])
        try:
            grid_lats = np.array(dset["lat"])
        except KeyError:
            grid_lats = np.array(dset["y"])
    mlons, mlats = np.meshgrid(grid_lons, grid_lats)
    mlons = np.deg2rad(mlons[~np.isnan(erosion)])
    mlats = np.deg2rad(mlats[~np.isnan(erosion)])
    erosion = erosion[~np.isnan(erosion)]
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
    
    erosion_col = np.full(df.shape[0], np.nan)
    
    for i in range(df.shape[0]):
        indices_point = radius_indices[i]
        
        # If no points within radius, use the nearest point
        if indices_point.size == 0:
            nearest_idx = nearest_indices[i][0]
            data = np.array([erosion[nearest_idx]])
        else:
            data = erosion[indices_point]
            
        # Calculate mean erosion
        erosion_col[i] = np.nanmean(data)
    
    # Add the single column with the new name
    df["erosion (m)"] = erosion_col
    
    return df


def clean_outliers(
    data: ArrayLike,
    contamination: Union[float, str] = "auto",
    random_state: Optional[int] = 42,
    return_mask: bool = False,
    **kwargs
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    
    # Input validation
    if data is None:
        raise ValueError("Input data cannot be None")
    
    data = np.array(data)
    
    if data.size == 0:
        if return_mask:
            return data, np.array([], dtype=bool)
        return data
    
    # Handle edge case: single data point
    if data.size == 1:
        if return_mask:
            return data, np.array([True])
        return data
    
    # Create Isolation Forest with fixed random state for reproducibility
    forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        **kwargs
    )
    
    # Reshape data for sklearn if needed
    if data.ndim == 1:
        x = data.reshape(-1, 1)
    else:
        x = data
    
    try:
        # Fit and predict
        forest_result = forest.fit_predict(x)
        
        # Create boolean mask for inliers
        inlier_mask = forest_result == 1
        
        # Filter data based on mask
        if data.ndim == 1:
            cleaned_data = data[inlier_mask]
        else:
            cleaned_data = data[inlier_mask, :]
        
        if return_mask:
            return cleaned_data, inlier_mask
        return cleaned_data
        
    except Exception as e:
        raise RuntimeError(f"Error during outlier detection: {str(e)}")


def clean_outliers_advanced(
    data: ArrayLike,
    method: str = "isolation_forest",
    contamination: Union[float, str] = "auto",
    random_state: Optional[int] = 42,
    return_mask: bool = False,
    **kwargs
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    
    data = np.array(data)
    
    if method == "isolation_forest":
        return clean_outliers(
            data, contamination=contamination, 
            random_state=random_state, return_mask=return_mask, **kwargs
        )
    
    elif method == "z_score":
        threshold = kwargs.get('threshold', 3.0)
        if data.ndim == 1:
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            inlier_mask = z_scores < threshold
        else:
            # For multivariate data, use Euclidean distance from center
            center = np.mean(data, axis=0)
            distances = np.sqrt(np.sum((data - center)**2, axis=1))
            threshold_val = np.mean(distances) + threshold * np.std(distances)
            inlier_mask = distances < threshold_val
            
    elif method == "iqr":
        if data.ndim == 1:
            Q1, Q3 = np.percentile(data, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            inlier_mask = (data >= lower_bound) & (data <= upper_bound)
        else:
            # Apply IQR to each column
            inlier_mask = np.ones(data.shape[0], dtype=bool)
            for col in range(data.shape[1]):
                Q1, Q3 = np.percentile(data[:, col], [25, 75])
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_mask = (data[:, col] >= lower_bound) & (data[:, col] <= upper_bound)
                inlier_mask &= col_mask
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply mask
    if data.ndim == 1:
        cleaned_data = data[inlier_mask]
    else:
        cleaned_data = data[inlier_mask, :]
    
    if return_mask:
        return cleaned_data, inlier_mask
    return cleaned_data
