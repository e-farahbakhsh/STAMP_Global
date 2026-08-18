'''
Preservation

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 11/08/2026
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
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
    
    """
    Coregister point data with cumulative erosion grids through geological time.

    This function matches a set of spatio-temporal point data (with present-day
    longitude, latitude, and geological age) to gridded cumulative erosion
    datasets stored as NetCDF files. For each time step, points are compared to
    nearby erosion grid cells using a haversine distance metric. If grid cells
    are found within the specified distance threshold (in degrees), the mean
    erosion value is assigned to the point. If no nearby cells are found, the
    nearest grid cell is used as a fallback. The resulting erosion values (in
    meters) are appended as a new column `"erosion (m)"` to the input data.

    Parameters
    ----------
    point_data : str, pandas.DataFrame, or path-like
        Input point dataset containing at least the columns:
        "age (Ma)", "present_lon", and "present_lat". If a string is
        provided, it is read as a CSV file.
    input_dir : str or path-like
        Directory containing cumulative erosion NetCDF files, expected to follow
        the naming convention "cumulative_erosion_{time}Ma.nc".
    distance_threshold : float, optional, default=0.1
        Maximum search radius (in degrees) around each point within which erosion
        grid cells are considered for averaging. If no cells are found, the nearest
        cell is used instead.
    output_filename : str or path-like, optional
        If provided, the resulting DataFrame is written to a CSV file at this
        location. Missing directories are created if necessary.
    n_jobs : int, optional, default=-2
        Number of parallel jobs to use for processing multiple time slices.
        Follows `joblib.Parallel` conventions (e.g., -1 uses all CPUs, -2 uses
        all but one).
    verbose : bool, optional, default=False
        If True, prints progress and file output information.

    Returns
    -------
    pandas.DataFrame
        The input dataset with an additional "erosion (m)" column containing
        mean erosion values from the grid data.

    Notes
    -----
    - Grid data are expected to contain dimensions "lon" and "lat" (or "x"
      and "y") and a variable "z" representing erosion in meters.
    - Points are grouped and processed by geological age for efficiency.
    - Uses haversine distances on a spherical Earth to match points to grid cells.
    """

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
    
    """
    Detect and remove statistical outliers from univariate or multivariate data
    using multiple configurable methods.

    This function extends `clean_outliers` by offering several techniques for
    outlier detection, including tree-based, distribution-based, and
    interquartile range (IQR) methods. It supports both one-dimensional and
    multi-dimensional data arrays.

    Parameters
    ----------
    data : array-like
        Input data, either 1D or 2D. If 2D, rows are treated as samples and
        columns as features.
    method : {"isolation_forest", "z_score", "iqr"}, default="isolation_forest"
        Outlier detection method:
        - `"isolation_forest"`: Uses sklearn's Isolation Forest algorithm for
          unsupervised anomaly detection.
        - `"z_score"`: Uses standard deviation thresholding to flag points
          far from the mean (univariate or multivariate with Euclidean distance).
        - `"iqr"`: Uses the interquartile range rule (1.5 × IQR) to filter
          outliers, applied per feature for multivariate data.
    contamination : float or "auto", optional
        Proportion of expected outliers (only used if `method="isolation_forest"`).
        If "auto", the algorithm automatically estimates the proportion.
    random_state : int, optional, default=42
        Seed for reproducibility (only used if `method="isolation_forest"`).
    return_mask : bool, default=False
        If True, also return the boolean mask indicating which samples are inliers.
    **kwargs : dict
        Additional keyword arguments passed to the selected method:
        - For `"z_score"`: `threshold` (float, default=3.0), the number of
          standard deviations to use as cutoff.
        - For `"isolation_forest"`: Any parameters supported by
          `sklearn.ensemble.IsolationForest`.
        - For `"iqr"`: No additional parameters required.

    Returns
    -------
    cleaned_data : ndarray
        Array containing only the inlier samples.
    inlier_mask : ndarray of bool, optional
        Boolean mask of the same length as the input data, where `True` indicates
        an inlier. Returned only if `return_mask=True`.

    Raises
    ------
    ValueError
        If `method` is not one of the supported options.
    RuntimeError
        If an error occurs during model fitting (only applies to isolation forest).

    Notes
    -----
    - For univariate data, all methods operate directly on the array.
    - For multivariate data:
      - "z_score" uses Euclidean distance from the mean vector.
      - "iqr" applies bounds independently to each feature.
      - "isolation_forest" uses the full multivariate distribution.
    - The "iqr" and "z_score" methods are deterministic, while
      "isolation_forest" may vary unless a fixed `random_state` is set.
    """
    
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


class PreservationScore:

    """
    Preservation-exposure score estimated by case-control logistic regression.

    Replaces a probability density fitted to the cumulative erosion of known
    deposits alone. A density of that kind, f(E | deposit), is not a
    preservation probability, because it partly reproduces how common each
    erosion value is across the available arc area. Comparing deposits against
    the unlabelled background instead recovers the relative enrichment,
    f(E | deposit) / f(E | background), which is proportional to
    P(deposit | E) up to a constant.

    The model is a cubic B-spline basis in log-transformed cumulative erosion
    followed by a regularised logistic regression. The returned score is the
    linear predictor rescaled to the unit interval over the fitted erosion
    range. Rescaling is monotonic, so location rankings are preserved, and the
    bounds are stored on the fitted object so that every point set scored with
    it shares one common scale.

    Points with zero cumulative erosion should be excluded from the fit.
    Cumulative erosion is identically zero at 0 Ma by construction, so the
    zero value carries no information about preservation and its extreme
    over-representation in the background would otherwise distort the fit.

    Parameters
    ----------
    n_knots : int, optional, default=6
        Number of knots in the spline basis. Values of 5 or 6 give a stable
        single-peaked curve. Larger values begin to fit noise at low erosion.
    degree : int, optional, default=3
        Degree of the spline basis.
    offset : float, optional, default=50.0
        Constant in metres added before log transformation, which keeps the
        transform finite at small erosion values.
    C : float, optional, default=1.0
        Inverse regularisation strength passed to the logistic regression.
        Smaller values give a smoother curve.

    Attributes
    ----------
    model_ : sklearn.pipeline.Pipeline
        The fitted spline and logistic regression pipeline.
    bounds_ : tuple of float
        Minimum and maximum of the linear predictor over the fitted data,
        used to rescale the score to the unit interval.

    Notes
    -----
    - The score is a bounded relative measure of preservation and exposure. It
      is not a calibrated probability that a deposit has survived.
    - No value is overridden. Zero erosion receives the score the fitted
      function assigns it, and heavily eroded ground receives a low score
      rather than being dropped.
    """

    def __init__(
        self,
        n_knots: int = 6,
        degree: int = 3,
        offset: float = 50.0,
        C: float = 1.0,
    ) -> None:
        self.n_knots = n_knots
        self.degree = degree
        self.offset = offset
        self.C = C
        self.model_ = None
        self.bounds_ = None

    def _transform(self, erosion: ArrayLike) -> NDArray:
        return np.log10(np.asarray(erosion, dtype=float) + self.offset).reshape(-1, 1)

    def fit(
        self,
        deposit_erosion: ArrayLike,
        background_erosion: ArrayLike,
    ) -> "PreservationScore":

        """
        Fit the score on known deposits against the unlabelled background.

        Parameters
        ----------
        deposit_erosion : array-like
            Cumulative erosion in metres at known deposit locations.
        background_erosion : array-like
            Cumulative erosion in metres at unlabelled background locations.

        Returns
        -------
        PreservationScore
            The fitted object.
        """

        deposit_erosion = np.asarray(deposit_erosion, dtype=float)
        background_erosion = np.asarray(background_erosion, dtype=float)

        erosion = np.concatenate([deposit_erosion, background_erosion])
        labels = np.concatenate(
            [np.ones(deposit_erosion.size), np.zeros(background_erosion.size)]
        )
        finite = np.isfinite(erosion)
        erosion, labels = erosion[finite], labels[finite]

        self.model_ = make_pipeline(
            SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                extrapolation="constant",
            ),
            LogisticRegression(C=self.C, max_iter=2000),
        ).fit(self._transform(erosion), labels)

        predictor = self.model_.decision_function(self._transform(erosion))
        self.bounds_ = (float(np.min(predictor)), float(np.max(predictor)))
        return self

    def score(self, erosion: ArrayLike) -> NDArray:

        """
        Return the preservation-exposure score on the unit interval.

        Parameters
        ----------
        erosion : array-like
            Cumulative erosion in metres.

        Returns
        -------
        ndarray
            Score values in the range zero to one, with NaN where the input
            erosion is not finite.
        """

        if self.model_ is None:
            raise RuntimeError("Call fit() before score().")

        erosion = np.asarray(erosion, dtype=float)
        out = np.full(erosion.shape, np.nan)
        finite = np.isfinite(erosion)
        lower, upper = self.bounds_
        predictor = self.model_.decision_function(self._transform(erosion[finite]))
        out[finite] = np.clip((predictor - lower) / (upper - lower), 0.0, 1.0)
        return out

    def held_out_auc(
        self,
        deposit_erosion: ArrayLike,
        background_erosion: ArrayLike,
        n_splits: int = 25,
        train_fraction: float = 0.7,
        random_state: Optional[int] = 42,
    ) -> Tuple[float, float]:

        """
        Evaluate the score on held-out deposits.

        The model is refitted on a random training fraction and scored on the
        remainder, repeated over several splits. Returns the mean and standard
        deviation of the area under the receiver operating characteristic curve.
        A value of 0.5 indicates no discrimination between deposits and
        background.
        """

        deposit_erosion = np.asarray(deposit_erosion, dtype=float)
        background_erosion = np.asarray(background_erosion, dtype=float)
        erosion = np.concatenate([deposit_erosion, background_erosion])
        labels = np.concatenate(
            [np.ones(deposit_erosion.size), np.zeros(background_erosion.size)]
        )
        rng = np.random.default_rng(random_state)

        def _auc(values, truth):
            order = np.argsort(values)
            truth = np.asarray(truth)[order]
            n_pos = truth.sum()
            n_neg = truth.size - n_pos
            ranks = np.arange(1, truth.size + 1)
            return (ranks[truth == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

        scores = []
        for _ in range(n_splits):
            index = rng.permutation(labels.size)
            cut = int(train_fraction * labels.size)
            train, test = index[:cut], index[cut:]
            fold = PreservationScore(
                n_knots=self.n_knots,
                degree=self.degree,
                offset=self.offset,
                C=self.C,
            ).fit(
                erosion[train][labels[train] == 1],
                erosion[train][labels[train] == 0],
            )
            scores.append(_auc(fold.score(erosion[test]), labels[test]))

        return float(np.mean(scores)), float(np.std(scores))

    def summary(self, grid: Optional[ArrayLike] = None) -> pd.DataFrame:

        """Tabulate the fitted score over a grid of cumulative erosion values."""

        if grid is None:
            grid = np.array(
                [0, 250, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 30000],
                dtype=float,
            )
        grid = np.asarray(grid, dtype=float)
        return pd.DataFrame({"erosion (m)": grid, "preservation score": self.score(grid)})
