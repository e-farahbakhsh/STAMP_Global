'''
Predictive Modelling

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 05/08/2025
'''

import os
from sys import stderr

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rio
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from sklearn.metrics import auc, roc_auc_score, roc_curve
import seaborn as sns
import xarray as xr


def roc_plot(y_test, z_test, n_classes, labels_name, average='macro'):
    
    fpr = {}
    tpr = {}
    roc_auc = {}

    y_test_dummies = pd.get_dummies(y_test).values
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], z_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='{}, AUC = {}'.format(labels_name[i], '{0:.4f}'.format(roc_auc[i])))
    
    ax.legend(loc='best')
    ax.grid(alpha=0.5)
    sns.despine()
    plt.show()
    print('ROC AUC score:', roc_auc_score(y_test_dummies, z_test, average=average))
    

def calculate_entropy(probabilities):
 
    # Ensure probabilities are within [0,1] range
    probabilities = np.clip(probabilities, 1e-15, 1-1e-15)
    
    # For binary classification, get both class probabilities
    p_class1 = probabilities
    p_class0 = 1 - p_class1
    
    # Entropy calculation: -sum(p_i * log2(p_i))
    entropy = -p_class0 * np.log2(p_class0) - p_class1 * np.log2(p_class1)
    
    return entropy


def calculate_tree_vote_variance(rf_model, X, n_jobs=-2):

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
