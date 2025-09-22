'''
Prospectivity Map

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 05/07/2025
'''


import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pygplates
from shapely.geometry import MultiPoint, Point
from sklearn.kernel_ridge import KernelRidge


def reconstruct_coastline_nodes(
    coastlines,
    resolution,
    polygons_dir,
    time_steps,
    plate_reconstruction,
    output_filename=None,
    verbose=False,
):
    
    # Load coastline data
    if not isinstance(coastlines, gpd.GeoDataFrame):
        coastlines = gpd.read_file(coastlines)
    polygons = coastlines.geometry.unary_union

    rotation_model = plate_reconstruction.rotation_model
    static_polygons = plate_reconstruction.static_polygons

    # Generate grid points across global extent
    lons = np.arange(-180, 180 + resolution, resolution)
    lats = np.arange(-90, 90 + resolution, resolution)
    mlons, mlats = np.meshgrid(lons, lats)
    coords = np.column_stack((mlons.ravel(), mlats.ravel()))
    mp = MultiPoint(coords)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        intersection = polygons.intersection(mp)

    if hasattr(intersection, "geoms"):
        coords = np.row_stack([g.coords for g in intersection.geoms])
    elif intersection.is_empty:
        coords = np.empty((0, 2))
    else:
        coords = np.array(intersection.coords)

    if coords.shape[0] == 0:
        raise ValueError("No intersection points found within coastlines.")

    # Save original coordinates (present-day)
    original_coords = {
        idx: {"present_lon": lon, "present_lat": lat}
        for idx, (lon, lat) in enumerate(coords)
    }

    # Create pygplates features from points
    features = []
    for idx, (lon, lat) in enumerate(coords):
        point = pygplates.PointOnSphere(float(lat), float(lon))
        feature = pygplates.Feature()
        feature.set_geometry(point)
        feature.set_valid_time(max(time_steps), min(time_steps))
        feature.set_name(str(idx))  # Save index as name
        features.append(feature)

    # Partition into plates
    partitioned = pygplates.partition_into_plates(
        partitioning_features=static_polygons,
        rotation_model=rotation_model,
        features_to_partition=features,
    )

    # Reconstruct features and store results
    data = []
    for time in time_steps:
        if verbose and time % 50 == 0:
            print(f"Reconstructing for time step {time} Ma...")

        # Load the polygon for the current time step
        geojson_filename = polygons_dir + f'/buffer_zones_{time}Ma.geojson'
        time_polygon_gdf = gpd.read_file(geojson_filename)
        time_polygon = time_polygon_gdf.geometry.unary_union  # Create a union of polygons if multiple

        reconstructed = []
        pygplates.reconstruct(partitioned, rotation_model, reconstructed, time)

        for r in reconstructed:
            lat, lon = r.get_reconstructed_geometry().to_lat_lon()
            index = int(r.get_feature().get_name())
            present = original_coords[index]

            # Check if the reconstructed point is inside the polygon for this time step
            point = Point(lon, lat)
            if point.within(time_polygon):
                data.append({
                    "index": index,
                    "lon": lon,
                    "lat": lat,
                    "age (Ma)": time,
                    "present_lon": present["present_lon"],
                    "present_lat": present["present_lat"],
                })

    df = pd.DataFrame(data)

    # Optionally save to CSV
    if output_filename:
        df.to_csv(output_filename, index=False)
        if verbose:
            print(f"Saved reconstructed data to {output_filename}")

    return df


def smooth_curve(X, y, alpha=0.1, gamma=1e-3, num_points=300):
    # Fit Kernel Ridge Regression with RBF (Gaussian) kernel
    # alpha: regularization strength (higher = smoother)
    # gamma: controls the width of the Gaussian kernel (smaller = smoother, broader influence)
    # Experiment with gamma=1e-3, 1e-4, 1e-5 and alpha=0.1, 1.0, 10.0 to get the right smoothness

    model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
    model.fit(X, y)
    
    X_smooth = np.linspace(X.min(), X.max(), num_points).reshape(-1, 1)
    y_smooth = model.predict(X_smooth)
    
    return X_smooth, y_smooth
