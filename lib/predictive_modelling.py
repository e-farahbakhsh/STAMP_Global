'''
Predictive Modelling

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 04/05/2026
'''

import os
from sys import stderr

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rio
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import auc, roc_auc_score, roc_curve
import seaborn as sns
import xarray as xr


def roc_plot(y_test, z_test, n_classes, labels_name, average='macro', fig_path=None):
    
    """
   Plot Receiver Operating Characteristic (ROC) curves for a multi-class classifier
   and report the Area Under the Curve (AUC) score.
   Parameters
   ----------
   y_test : array-like, shape (n_samples,)
       True class labels for each sample.
   z_test : array-like, shape (n_samples, n_classes)
       Predicted probabilities or decision scores for each class.
   n_classes : int
       Number of unique classes in the classification problem.
   labels_name : list of str
       Names of the classes, used as labels in the plot legend.
   average : {'macro', 'micro', 'weighted'}, default='macro'
       Type of averaging to use when computing the overall ROC AUC score.
   Returns
   -------
   None
       Displays a ROC curve plot with one curve per class and prints the overall
       ROC AUC score.
   Notes
   -----
   - The function first converts `y_test` into one-hot encoded form.
   - For each class, it computes the false positive rate (FPR), true positive rate (TPR),
     and the AUC, then plots the corresponding ROC curve.
   - A diagonal line (y = x) is drawn as a reference for random classification.
   - The plot includes class-specific ROC curves, each annotated with its AUC value.
   - Prints the global ROC AUC score based on the specified averaging method.
   """
    
    # --- fontsize controls (tweak as needed) ---
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12
    title_fontsize = 16
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    y_test_dummies = pd.get_dummies(y_test).values
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], z_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # roc for each class
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("whitesmoke")
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors[0] = 'lightsalmon'   # was tab:blue
    colors[1] = 'darkseagreen'    # was tab:orange
    ax.set_prop_cycle(color=colors)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False positive rate', fontsize=label_fontsize)
    ax.set_ylabel('True positive rate', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    # ax.set_title('Receiver Operating Characteristic', fontsize=title_fontsize)
    
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], lw=2, label='{}, AUC = {}'.format(labels_name[i], '{0:.4f}'.format(roc_auc[i])))
    
    ax.legend(loc='best', fontsize=legend_fontsize)
    ax.grid(alpha=0.5)
    sns.despine()
    
    plt.tight_layout()
    
    if fig_path is not None:
        if not os.path.exists(fig_path):
            fig.savefig(
                fig_path,
                dpi=300,
                bbox_inches="tight"
            )
    
    plt.show()
    
    print('ROC AUC score:', roc_auc_score(y_test_dummies, z_test, average=average))
    

def calculate_entropy(probabilities):
    
    """
    Compute the Shannon entropy for binary classification probabilities.

    Parameters
    ----------
    probabilities : array-like or float
        Predicted probability of the positive class (class 1). Can be a single
        float or an array of values in the range [0, 1].

    Returns
    -------
    entropy : array-like or float
        Entropy value(s), measuring the uncertainty of the prediction(s).
        - 0 indicates no uncertainty (probability close to 0 or 1).
        - Maximum entropy (1 bit) occurs when probabilities are 0.5 (maximum uncertainty).

    Notes
    -----
    - Probabilities are clipped to avoid numerical issues when computing log2(0).
    - The entropy is calculated using the Shannon formula:
      
          H(p) = -[p * log2(p) + (1 - p) * log2(1 - p)]
    """
 
    # Ensure probabilities are within [0,1] range
    probabilities = np.clip(probabilities, 1e-15, 1-1e-15)
    
    # For binary classification, get both class probabilities
    p_class1 = probabilities
    p_class0 = 1 - p_class1
    
    # Entropy calculation: -sum(p_i * log2(p_i))
    entropy = -p_class0 * np.log2(p_class0) - p_class1 * np.log2(p_class1)
    
    return entropy


def calculate_tree_vote_variance(rf_model, X, n_jobs=-2):
    
    """
    Calculate the variance of tree-level predictions (votes) in a Random Forest model.

    Parameters
    ----------
    rf_model : sklearn.ensemble.RandomForestClassifier or RandomForestRegressor
        A trained Random Forest model with an attribute `estimators_` containing
        the individual decision trees.

    X : array-like, shape (n_samples, n_features)
        Input data on which to compute predictions from each tree.

    n_jobs : int, default=-2
        Number of parallel jobs to run when collecting predictions.
        - `-1` uses all available cores.
        - `-2` uses all but one core.

    Returns
    -------
    vote_variance : ndarray, shape (n_samples,)
        Variance of predictions across all trees for each sample.
        - Higher variance indicates less agreement among trees (more uncertainty).
        - Lower variance indicates stronger consensus among trees.

    Notes
    -----
    - For classification, each tree outputs a class label (hard vote).
    - For regression, each tree outputs a numeric prediction.
    - This metric can be used as an uncertainty measure to identify samples where
      the Random Forest model is less confident.
    """

    n_trees = len(rf_model.estimators_)
    
    # Function to get predictions from a single tree
    def get_tree_predictions(tree_idx):
        
        tree = rf_model.estimators_[tree_idx]
        
        return tree.predict(X)
    
    tree_indices = range(n_trees)
    
    # Run tree predictions in parallel
    tree_predictions = Parallel(n_jobs=n_jobs)(
        delayed(get_tree_predictions)(i) for i in tree_indices
    )
    
    # Convert list of predictions to a matrix [n_samples, n_trees]
    all_tree_preds = np.column_stack(tree_predictions)
    # Calculate variance of votes for each sample
    vote_variance = np.var(all_tree_preds, axis=1)
    
    return vote_variance


def create_grids(
    data,
    output_dir,
    times,
    resolution=None,
    extent=None,
    interpolation=False,
    threads=1,
    verbose=False,
    column="probability",
    filename_format="probability_grid_{}Ma.nc",
):
    
    """
    Generate and save gridded NetCDF datasets from point-based spatio-temporal data.

    This function takes scattered data points (longitude, latitude, values) for
    different geological times, interpolates them onto regular grids (or directly
    maps to nearest grid nodes), and saves each time slice as a NetCDF file.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Input dataset containing at least the following columns:
        - 'age (Ma)' : geological time in millions of years.
        - 'lon'      : longitude coordinates of points.
        - 'lat'      : latitude coordinates of points.
        - `column`   : the variable to be gridded (default: "probability").
        If a string is given, it is treated as a path to a CSV file.

    output_dir : str
        Directory where output NetCDF grid files will be saved.

    times : list of int or float
        List of geological times (Ma) at which to generate grids.

    resolution : float, optional
        Grid spacing in degrees. If None, it is estimated from the input data.

    extent : tuple or str, optional
        Spatial extent of the grid. Can be:
        - None: bounding box is inferred from the data.
        - "global": full extent (-180, 180, -90, 90).
        - tuple (xmin, xmax, ymin, ymax): custom bounding box.

    interpolation : bool, default=False
        If True, interpolate values onto the grid using `scipy.interpolate.griddata`.
        If False, assign values only at exact data locations.

    threads : int, default=1
        Number of parallel workers. If >1, time steps are processed in parallel.

    verbose : bool, default=False
        If True, print progress information during processing.

    column : str, default="probability"
        Column in `data` to be gridded.

    filename_format : str, default="probability_grid_{}Ma.nc"
        Format string for output filenames. The placeholder `{}` will be replaced
        with the rounded integer geological time (Ma).

    Returns
    -------
    None
        Saves one NetCDF file per time step to `output_dir`.
        Each file contains:
        - Coordinates: latitude (`lat`), longitude (`lon`)
        - Variable: `z` (gridded values from `column`)

    Notes
    -----
    - Grids are saved in WGS84 geographic coordinates (EPSG:4326).
    - If interpolation is enabled, distant grid nodes are masked (set to NaN)
      if further than one grid cell from the nearest data point.
    - Output is compressed (`zlib=True`) to reduce file size.
    """

    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        data = pd.DataFrame(data)

    data = data.dropna(subset=[column])

    if threads == 1:
        for time in times:
            _create_grid_time(
                time=time,
                data_lons=np.array(data[data["age (Ma)"] == time]["lon"]),
                data_lats=np.array(data[data["age (Ma)"] == time]["lat"]),
                data_values=np.array(data[data["age (Ma)"] == time][column]),
                resolution=resolution,
                output_dir=output_dir,
                extent=extent,
                verbose=verbose,
                filename_format=filename_format,
            )
    else:
        with Parallel(threads, verbose=10 * int(verbose)) as p:
            p(
                delayed(_create_grid_time)(
                    time=time,
                    data_lons=np.array(data[data["age (Ma)"] == time]["lon"]),
                    data_lats=np.array(data[data["age (Ma)"] == time]["lat"]),
                    data_values=np.array(data[data["age (Ma)"] == time][column]),
                    resolution=resolution,
                    output_dir=output_dir,
                    extent=extent,
                    interpolation=interpolation,
                    verbose=False,
                    filename_format=filename_format,
                )
                for time in times
            )
            

def _create_grid_time(
    time,
    data_lons,
    data_lats,
    data_values,
    resolution,
    output_dir,
    extent=None,
    interpolation=False,
    verbose=False,
    filename_format="probability_grid_{}Ma.nc",
):
    
    time = int(np.around(time))
    output_filename = os.path.join(
        output_dir, filename_format.format(time)
    )

    # Determine extent
    if extent is None:
        xmin = np.nanmin(data_lons)
        xmax = np.nanmax(data_lons)
        ymin = np.nanmin(data_lats)
        ymax = np.nanmax(data_lats)
    elif extent == "global":
        xmin, xmax, ymin, ymax = -180, 180, -90, 90
    else:
        xmin, xmax, ymin, ymax = extent

    # Determine resolution
    if resolution is None:
        resx = np.nanmin(np.gradient(np.sort(np.unique(data_lons))))
        resy = np.nanmin(np.gradient(np.sort(np.unique(data_lats))))
    else:
        resx = resolution
        resy = resolution

    # Create the grid
    grid_lons = np.arange(xmin, xmax + resx, resx)
    grid_lats = np.arange(ymin, ymax + resy, resy)
    grid_mlons, grid_mlats = np.meshgrid(grid_lons, grid_lats)

    if interpolation:
        # Interpolate
        points = np.column_stack((data_lons, data_lats))
        arr = griddata(points, data_values, (grid_mlons, grid_mlats), method="nearest")  # or "linear", "cubic"
        
        # Mask distant grid nodes
        tree = cKDTree(points)
        flat_grid_points = np.column_stack((grid_mlons.ravel(), grid_mlats.ravel()))
        distances, _ = tree.query(flat_grid_points, k=1)
        
        mask = distances > resolution
        arr.ravel()[mask] = np.nan  # Mask out distant nodes        
    else:
        arr = np.full((grid_lats.size, grid_lons.size), np.nan, dtype=float)
        for data_lon, data_lat, data_value in zip(
            data_lons, data_lats, data_values
        ):
            mask = np.logical_and(grid_mlons == data_lon, grid_mlats == data_lat)
            arr[mask] = data_value

    # Create the dataset
    dset = xr.Dataset(
        data_vars={
            "z": (("lat", "lon"), arr),
        },
        coords={
            "lon": grid_lons,
            "lat": grid_lats,
            # "time": time,
        },
    )
    
    # Set projection info
    dset.rio.write_crs(4326, inplace=True)
    
    if verbose:
        print(
            "\t- Writing output file: " + os.path.basename(output_filename),
            file=stderr,
        )
        
    # Save as NetCDF
    dset.to_netcdf(
        output_filename,
        encoding={
            "z": {
                "zlib": True,
                "dtype": "float32",
            }
        },
    )
    
    return dset


def smooth_curve(
    X,
    y,
    method="krr",
    num_points=300,
    # --- Kernel Ridge Regression (RBF) params ---
    alpha=0.001,
    gamma=0.001,
    # --- Gaussian smoothing params ---
    fwhm_myr=20.0,
    dt_myr=1.0,
    mode="nearest",
    truncate=4.0,
    enforce_uniform_dt=True,
):
    
    """
    Smooth a curve using either Kernel Ridge Regression (RBF)
    or Gaussian low-pass filtering.

    Parameters
    ----------
    X : array-like
        Independent variable (e.g., time).
    y : array-like
        Dependent variable to smooth.
    method : {"krr", "gaussian"}
        Smoothing method:
        - "krr"      : Kernel Ridge Regression with RBF kernel
        - "gaussian" : Gaussian low-pass filter
    num_points : int
        Number of points for resampling (only used for method="krr").

    KRR parameters
    --------------
    alpha : float
        Regularization strength (higher = smoother).
    gamma : float
        RBF kernel width (smaller = smoother).

    Gaussian parameters
    -------------------
    fwhm_myr : float
        Gaussian filter Full Width at Half Maximum (in same units as X).
    dt_myr : float
        Sampling interval of X.
    mode : str
        Boundary handling for Gaussian filter.
    truncate : float
        Truncate Gaussian at this many standard deviations.
    enforce_uniform_dt : bool
        If True, checks that X spacing matches dt_myr.

    Returns
    -------
    X_out : np.ndarray
        Smoothed X axis (resampled for KRR, original for Gaussian).
    y_smooth : np.ndarray
        Smoothed y values.
    """

    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}.")

    method = method.lower()

    # ------------------------------------------------------------------
    # Kernel Ridge Regression smoothing
    # ------------------------------------------------------------------
    if method == "krr":
        X_2d = X.reshape(-1, 1)

        model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
        model.fit(X_2d, y)

        X_smooth = np.linspace(X.min(), X.max(), num_points).reshape(-1, 1)
        y_smooth = model.predict(X_smooth)

        return X_smooth.ravel(), y_smooth

    # ------------------------------------------------------------------
    # Gaussian smoothing
    # ------------------------------------------------------------------
    elif method == "gaussian":
        if enforce_uniform_dt:
            dX = np.diff(X)
            if not np.allclose(dX, dt_myr, rtol=0, atol=1e-8):
                raise ValueError(
                    "X is not uniformly sampled at dt_myr. "
                    "Either resample X or set enforce_uniform_dt=False."
                )

        # Convert FWHM -> sigma (in samples)
        sigma_myr = fwhm_myr / 2.355
        sigma_samples = sigma_myr / dt_myr

        y_smooth = gaussian_filter1d(
            y,
            sigma=sigma_samples,
            mode=mode,
            truncate=truncate,
        )

        return X, y_smooth

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'krr' or 'gaussian'."
        )


def compute_spread(x, method="std", scale=False):
    if method == "std":
        return x.std()
    elif method == "mad":
        mad = np.median(np.abs(x - np.median(x)))
        return mad * 1.4826 if scale else mad  # scaled MAD ≈ std
    else:
        raise ValueError("method must be 'std' or 'mad'")
