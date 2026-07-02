'''
Prospectivity Map

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 05/05/2026
'''


import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pygplates
from shapely.geometry import MultiPoint, Point

from .reconstruction_coregistration import prepare_polygon_features, reconstruct_polygons


def reconstruct_coastline_nodes(
    coastlines,
    coastlines_recon,
    resolution,
    buffer_zones_dir,
    time_steps,
    plate_reconstruction,
    anchor_plate_id=None,
    coastlines_recon_dir=None,
    coastlines_recon_file="coastlines",
    output_filename=None,
    verbose=False,
):
    
    """
    Reconstruct continental grid nodes through geological time and filter them by
    continental landmass and trench buffer zone overlap.
    
    This function generates a uniform global grid at a specified spatial
    resolution and selects nodes that intersect present-day continental
    landmasses. These nodes are partitioned into tectonic plates using static
    plate polygons and then reconstructed through a series of geological time
    steps using a plate rotation model.
    
    At each reconstruction time, the function:
    
    1. Reconstructs continental landmass polygons.
    2. Reconstructs the continental grid nodes.
    3. Loads trench buffer zone geometries corresponding to that time step.
    4. Retains only nodes that fall within both the reconstructed continental
       polygons and the trench buffer zones.
    
    The resulting dataset represents continental locations that are proximal to
    subduction zones through time, which can be used to predict mineralisation probability.
    
    Parameters
    ----------
    coastlines : str or geopandas.GeoDataFrame
        Path to a present-day coastline shapefile/GeoJSON, or a GeoDataFrame
        containing coastline geometries used to identify continental grid nodes.
    
    coastlines_recon : str or pygplates.FeatureCollection
        GPlates-compatible feature file (e.g., GPML or SHP) containing present-day
        continental landmass polygons used for reconstruction.
    
    resolution : float
        Grid spacing in degrees used to sample the global grid
        (commonly 0.25°–1.0°).
    
    buffer_zones_dir : str
        Directory containing trench buffer zone GeoJSON files named by
        geological age (e.g., `buffer_zones_100Ma.geojson`).
    
    time_steps : sequence of float
        Geological times (Ma) at which continental nodes will be reconstructed.
    
    plate_reconstruction : gplately.PlateReconstruction
        Plate reconstruction object providing the rotation model and static
        polygons for plate partitioning.
        
    anchor_plate_id : int
        Anchor plate ID for reconstruction.
    
    coastlines_recon_dir : str, optional
        Directory used to store reconstructed continental polygon shapefiles.
        Files are reused if already present.
    
    output_filename : str, optional
        If provided, the reconstructed node dataset will be written to this CSV file.
    
    verbose : bool, default=False
        If True, prints progress messages during reconstruction.
    
    Returns
    -------
    pandas.DataFrame
        Table of reconstructed continental nodes with columns:
    
        - index : unique grid node identifier
        - lon, lat : reconstructed longitude and latitude (degrees)
        - age (Ma) : geological reconstruction time
        - present_lon, present_lat : original present-day coordinates
    
    Notes
    -----
    - Grid nodes are reconstructed using `pygplates.reconstruct`.
    - Spatial filtering requires nodes to lie within both reconstructed
      continental landmasses and the trench buffer zones for each time step.
    """

    rotation_model = plate_reconstruction.rotation_model
    static_polygons = plate_reconstruction.static_polygons

    if coastlines_recon_dir is None:
        coastlines_recon_dir = "reconstructed_coastlines"

    os.makedirs(coastlines_recon_dir, exist_ok=True)

    continents_fc = prepare_polygon_features(
        coastlines_recon,
        rotation_model,
        static_polygons,
        verbose,
    )

    # Load coastline geometry
    if not isinstance(coastlines, gpd.GeoDataFrame):
        coastlines = gpd.read_file(coastlines)

    coastline_union = coastlines.geometry.unary_union

    # Build grid
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
        raise ValueError("No grid points intersect present-day coastlines.")

    original_coords = {
        idx: {"present_lon": float(lon), "present_lat": float(lat)}
        for idx, (lon, lat) in enumerate(coords)
    }

    # Convert to pygplates features
    features = []

    t_min = float(min(time_steps))
    t_max = float(max(time_steps))

    for idx, (lon, lat) in enumerate(coords):

        pt = pygplates.PointOnSphere(float(lat), float(lon))

        f = pygplates.Feature()
        f.set_geometry(pt)
        f.set_valid_time(t_max, t_min)
        f.set_name(str(idx))

        features.append(f)

    partitioned = pygplates.partition_into_plates(
        partitioning_features=static_polygons,
        rotation_model=rotation_model,
        features_to_partition=features,
    )

    # Reconstruction loop
    out_rows = []

    for time in time_steps:

        if verbose and time % 100 == 0:
            print(f"Processing {time} Ma")

        buffer_path = os.path.join(
            buffer_zones_dir,
            f"buffer_zones_{int(time)}Ma.geojson",
        )

        buffer_union = gpd.read_file(buffer_path).geometry.unary_union

        continents_union = reconstruct_polygons(
            time,
            continents_fc,
            rotation_model,
            coastlines_recon_dir,
            coastlines_recon_file,
            anchor_plate_id,
        )

        reconstructed = []

        pygplates.reconstruct(
            partitioned,
            rotation_model,
            reconstructed,
            float(time),
            anchor_plate_id
        )

        for r in reconstructed:

            lat, lon = r.get_reconstructed_geometry().to_lat_lon()
            idx = int(r.get_feature().get_name())

            present = original_coords[idx]

            p = Point(lon, lat)

            if p.within(buffer_union) and p.within(continents_union):

                out_rows.append(
                    {
                        "index": idx,
                        "lon": lon,
                        "lat": lat,
                        "age (Ma)": float(time),
                        "present_lon": present["present_lon"],
                        "present_lat": present["present_lat"],
                    }
                )

    df = pd.DataFrame(out_rows)

    if output_filename:
        df.to_csv(output_filename, index=False)

    return df
