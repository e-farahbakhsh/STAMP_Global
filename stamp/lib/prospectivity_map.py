'''
Prospectivity Map

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 16/09/2025
'''


import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pygplates
from shapely.geometry import MultiPoint, Point
from sklearn.kernel_ridge import KernelRidge


def reconstruct_coastline_nodes(
    coastlines,
    coastlines_recon,
    resolution,
    buffer_zones_dir,
    time_steps,
    plate_reconstruction,
    coastlines_recon_dir=None,
    output_filename=None,
    verbose=False,
):
    
    """
    Reconstruct continental grid nodes through geological time and filter them by continental
    and trench buffer zone overlap.

    This function generates a uniform global grid at the specified spatial resolution,
    selects points that intersect present-day continental landmasses, assigns plate IDs to those
    points using static polygons, and reconstructs them through the given geological
    time steps using a GPlately PlateReconstruction model. At each reconstruction time,
    it retains only those points that fall within both the reconstructed continental
    landmass polygons and user-supplied trench buffer zone geometries.

    Parameters
    ----------
    coastlines : str or geopandas.GeoDataFrame
        Path to a coastline shapefile/GeoJSON, or a GeoDataFrame containing present-day
        coastline geometries.
    coastlines_recon : str
        Path to a GPlates-compatible feature file (e.g., GPML or SHP) containing
        present-day continental landmass polygons for the plate model.
    resolution : float
        Grid spacing in degrees used to sample the global grid (typically 0.25–1.0°).
    buffer_zones_dir : str
        Directory containing trench buffer zone GeoJSON files named by geological age,
        e.g. `buffer_zones_100Ma.geojson`.
    time_steps : list[float]
        Geological times (in Ma) to reconstruct and sample the continental nodes.
    plate_reconstruction : gplately.PlateReconstruction
        GPlately PlateReconstruction object providing `rotation_model` and
        `static_polygons` for plate partitioning and motion.
    coastlines_recon_dir : str, optional
        Directory to write reconstructed continental landmass shapefiles (e.g., `coastlines_100Ma.shp`).
        Created if it does not exist. Defaults to a subdirectory of the working directory.
    output_filename : str, optional
        Path to a CSV file for saving the reconstructed node coordinates and metadata.
    verbose : bool, default=False
        If True, prints progress messages during reconstruction and partitioning.

    Returns
    -------
    pandas.DataFrame
        Table of reconstructed continental nodes with columns:
        - `index`: Unique point index
        - `lon`, `lat`: Reconstructed longitude/latitude (degrees)
        - `age (Ma)`: Reconstruction time
        - `present_lon`, `present_lat`: Original (present-day) coordinates

    Notes
    -----
    - Plate IDs are assigned to each grid node using `pygplates.partition_into_plates()`
      with the static polygon layer.
    - Continental polygons lacking plate IDs are automatically partitioned before
      reconstruction.
    - Reconstructed continental landmasses are written to disk if missing, using
      `pygplates.reconstruct()`.
    - Points are filtered spatially to those within both reconstructed continental
      polygons and the trench buffer zones for each time step.
    """

    rotation_model = plate_reconstruction.rotation_model
    static_polygons = plate_reconstruction.static_polygons

    if not os.path.exists(coastlines_recon_dir):
        os.makedirs(coastlines_recon_dir, exist_ok=True)

    # Load coastline geometry and get unary union
    if not isinstance(coastlines, gpd.GeoDataFrame):
        coastlines = gpd.read_file(coastlines)
    coastline_union = coastlines.geometry.unary_union

    # Build a present-day grid and keep only nodes that intersect coastlines
    lons = np.arange(-180, 180 + resolution, resolution)
    lats = np.arange(-90, 90 + resolution, resolution)
    mlons, mlats = np.meshgrid(lons, lats)
    candidate_xy = np.column_stack((mlons.ravel(), mlats.ravel()))
    mp = MultiPoint(candidate_xy)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        intersection = coastline_union.intersection(mp)

    if hasattr(intersection, "geoms"):
        coords = np.row_stack([g.coords for g in intersection.geoms])
    elif getattr(intersection, "is_empty", False):
        coords = np.empty((0, 2))
    else:
        coords = np.array(intersection.coords)

    if coords.shape[0] == 0:
        raise ValueError("No grid points intersect present-day coastlines at the chosen resolution.")

    # Store original (present-day) coordinates
    original_coords = {idx: {"present_lon": float(lon), "present_lat": float(lat)}
                       for idx, (lon, lat) in enumerate(coords)}

    # Make pygplates Features from the points and set validity time
    features = []
    t_min, t_max = float(min(time_steps)), float(max(time_steps))
    for idx, (lon, lat) in enumerate(coords):
        pt = pygplates.PointOnSphere(float(lat), float(lon))
        f = pygplates.Feature()
        f.set_geometry(pt)
        # pyGPlates expects (oldest, youngest)
        f.set_valid_time(t_max, t_min)
        f.set_name(str(idx))  # stash the index
        features.append(f)

    # Partition the points into plates
    partitioned = pygplates.partition_into_plates(
        partitioning_features=static_polygons,
        rotation_model=rotation_model,
        features_to_partition=features,
    )

    # Continental landmass: ensure features have plate IDs and reconstruct per time to SHP
    continents_fc = pygplates.FeatureCollection(coastlines_recon)

    def _needs_plate_ids(fc):
        for feat in fc:
            if feat.get_reconstruction_plate_id() is None:
                return True
        return False

    if _needs_plate_ids(continents_fc):
        if verbose:
            print("Partitioning continental landmasses into plates (missing plate IDs)...")
        static_fc = (static_polygons
                     if isinstance(static_polygons, pygplates.FeatureCollection)
                     else pygplates.FeatureCollection(static_polygons))
        partitioned_conts = []
        pygplates.partition_into_plates(
            static_fc, rotation_model, continents_fc, partitioned_conts,
            properties_to_copy=pygplates.PartitionIntoPlatesCopyProperties.all,
            reconstruction_time=0.0,
        )
        continents_fc = pygplates.FeatureCollection(partitioned_conts)
        if verbose:
            print(f"Partitioned {len(continents_fc)} continent features.")

    # Helper: ensure continent SHP exists for a time, return its unary union polygon
    def _get_continent_union_for_time(t):
        shp_path = os.path.join(coastlines_recon_dir, f"coastlines_{int(t)}Ma.shp")
        if not os.path.exists(shp_path):
            # Write directly to SHP; SHP will split polygons at ±180°
            pygplates.reconstruct(continents_fc, rotation_model, shp_path, float(t))
        gdf = gpd.read_file(shp_path)
        gdf["geometry"] = gdf["geometry"].buffer(0) # Fix invalid polygons
        return gdf.geometry.unary_union

    # Reconstruct and filter
    out_rows = []

    for time in time_steps:
        if verbose and (int(time) % 50 == 0):
            print(f"Reconstructing points and filtering at {time} Ma ...")

        # 1) Load trench buffer zone polygon for this time
        buffer_path = os.path.join(buffer_zones_dir, f"buffer_zones_{int(time)}Ma.geojson")
        buffer_gdf = gpd.read_file(buffer_path)
        buffer_union = buffer_gdf.geometry.unary_union

        # 2) Load (or build) reconstructed continental landmasses for this time
        continents_union = _get_continent_union_for_time(time)

        # 3) Reconstruct the partitioned point features to this time
        reconstructed = []
        pygplates.reconstruct(partitioned, rotation_model, reconstructed, float(time))

        # 4) Keep only points inside BOTH polygons: (a) trench buffer zones AND (b) continents
        for r in reconstructed:
            lat, lon = r.get_reconstructed_geometry().to_lat_lon()
            idx = int(r.get_feature().get_name())
            present = original_coords[idx]

            p = Point(lon, lat)
            if p.within(buffer_union) and p.within(continents_union):
                out_rows.append({
                    "index": idx,
                    "lon": lon,
                    "lat": lat,
                    "age (Ma)": float(time),
                    "present_lon": present["present_lon"],
                    "present_lat": present["present_lat"],
                })

    df = pd.DataFrame(out_rows)

    # Optional save
    if output_filename:
        df.to_csv(output_filename, index=False)
        if verbose:
            print(f"Saved reconstructed data to {output_filename}")

    return df


def smooth_curve(X, y, alpha=0.001, gamma=0.001, num_points=300):
    # Fit Kernel Ridge Regression with RBF (Gaussian) kernel
    # alpha: regularization strength (higher = smoother)
    # gamma: controls the width of the Gaussian kernel (smaller = smoother, broader influence)
    # Experiment with gamma=1e-3, 1e-4, 1e-5 and alpha=0.1, 1.0, 10.0 to get the right smoothness

    model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
    model.fit(X, y)
    
    X_smooth = np.linspace(X.min(), X.max(), num_points).reshape(-1, 1)
    y_smooth = model.predict(X_smooth)
    
    return X_smooth, y_smooth
