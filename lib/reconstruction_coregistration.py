'''
Reconstruction and Coregistration

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 04/05/2026
'''


import concurrent.futures
from multiprocessing import cpu_count
import os
from sys import stderr
from typing import (
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)
import warnings

import geopandas as gpd
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from pathlib import Path
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import linemerge
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsRegressor
import xarray as xr

import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    EARTH_RADIUS,
    reverse_reconstruct_points_impl,
    reconstruct_points_impl,
)
from gplately.geometry import (
    pygplates_to_shapely,
    wrap_geometries,
)
from gplately.tools import xyz2lonlat


_PathLike = Union[os.PathLike, str]
_PathOrDataFrame = Union[_PathLike, pd.DataFrame]
_FeatureCollectionInput = Union[
    pygplates.FeatureCollection,
    str,
    pygplates.Feature,
    Iterable[pygplates.Feature],
    Iterable[
        Union[
            pygplates.FeatureCollection,
            str,
            pygplates.Feature,
            Iterable[pygplates.Feature],
        ]
    ],
]
_RotationModelInput = Union[
    pygplates.RotationModel,
    _FeatureCollectionInput,
]


def run_create_buffer_zones(
    times: Sequence[float],
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    static_polygons: Optional[_FeatureCollectionInput] = None,
    anchor_plate_id: int = None,
    target_extent: Optional[_PathLike] = None,
    output_dir: _PathLike = os.curdir,
    buffer_distance: float = 6,
    clip_to_overriding_plate: bool = False,
    n_jobs: int = -2,
    verbose: bool = False,
    return_output: bool = False,
) -> Optional[List[gpd.GeoDataFrame]]:
    
    """
    Generate buffer zones around subduction trenches for a sequence of geological
    times using a plate reconstruction model.
    
    This function automates the creation of one-sided buffer polygons around
    subduction trenches, which can be used to approximate arc–backarc
    environments. It supports parallel execution across multiple time steps.
    
    Parameters
    ----------
    times : Sequence[float]
        Geological times (in Ma) at which buffer zones should be generated.
    plate_reconstruction : PlateReconstruction, optional
        A pre-initialised GPlately PlateReconstruction object. If not provided,
        `rotation_model`, `topology_features`, and `static_polygons` must be supplied.
    rotation_model : str, Path, pygplates.RotationModel, or FeatureCollection, optional
        Rotation model used for plate kinematics if no PlateReconstruction is provided.
    topology_features : str, Path, FeatureCollection, or sequence, optional
        Topological plate boundary features (e.g., trenches, ridges, transforms).
    static_polygons : str, Path, FeatureCollection, or sequence, optional
        Present-day static polygons used to assign plate IDs.
    anchor_plate_id : int
        Anchor plate ID for reconstruction.
    target_extent: _PathLike: default=None
        Path to the desired polygon to clip the buffer zones to its sxtent
    output_dir : str or Path-like, default=os.curdir
        Directory where buffer zone files (GeoJSON) will be saved.
    buffer_distance : float, default=6
        Buffer distance in degrees (converted internally to metres) to extend
        from trench lines, applied one-sided based on subduction polarity.
    clip_to_overriding_plate: bool = False,
        If True, clip the buffer zones to the overriding plates
    n_jobs : int, default=-2
        Number of parallel processes. Follows joblib semantics (-1 for all CPUs,
        -2 for all but one, etc.).
    verbose : bool, default=False
        Print progress and diagnostic information if True.
    return_output : bool, default=False
        If True, return a list of GeoDataFrames containing buffer zones for each
        time step. If False, results are only written to file.
    
    Returns
    -------
    list of geopandas.GeoDataFrame, optional
        Only returned if `return_output=True`. Each GeoDataFrame contains buffer
        polygons and associated plate/subduction attributes at a given time step.
    
    Notes
    -----
    - Buffer zones are generated using great-circle projection and are oriented
      according to subduction polarity (left vs right).
    - If `clip_to_overriding_plate=True` is enabled downstream, buffers are clipped
      to the overriding plate polygons to restrict them to trench-adjacent regions.
    - Output files are named `buffer_zones_{time}Ma.geojson`.
    """
    
    if plate_reconstruction is None:
        if topology_features is None or rotation_model is None:
            raise TypeError(
                "Either `plate_reconstruction` or both "
                "`topology_features` and `rotation_model` "
                "must not be None."
            )

    if output_dir is not None and not os.path.isdir(output_dir):
        if verbose:
            print(
                "Output directory does not exist; creating now: "
                + output_dir,
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    times_split = np.array_split(times, n_jobs)
    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        results = parallel(
            delayed(_multiple_timesteps_buffer)(
                times=t,
                buffer_distance=buffer_distance,
                return_output=return_output,
                plate_reconstruction=plate_reconstruction,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
                target_extent=target_extent,
                output_dir=output_dir,
                clip_to_overriding_plate=clip_to_overriding_plate,
            )
            for t in times_split
        )
        
    if return_output:
        out = []
        for i in results:
            out.extend(i)
        return out
    
    return None


def _multiple_timesteps_buffer(
    times: Sequence[float],
    buffer_distance: float,
    return_output: bool,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    static_polygons: Optional[_FeatureCollectionInput] = None,
    anchor_plate_id: int = None,
    target_extent: Optional[_PathLike] = None,
    output_dir: _PathLike = os.curdir,
    clip_to_overriding_plate: bool = False,
):
    
    if plate_reconstruction is None:
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)
        if not isinstance(topology_features, pygplates.FeatureCollection):
            topology_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(topology_features).get_features()
                )
        if not isinstance(static_polygons, pygplates.FeatureCollection):
            static_polygons = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(static_polygons).get_features()
                )

    out = []
    for time in times:
        out.append(
            _create_buffer_zones(
                time=time,
                plate_reconstruction=plate_reconstruction,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
                target_extent=target_extent,
                output_dir=output_dir,
                buffer_distance=buffer_distance,
                clip_to_overriding_plate=clip_to_overriding_plate,
                return_output=return_output,
            )
        )
        
    if return_output:
        return out


def _create_buffer_zones(
    time: float,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    static_polygons: Optional[_FeatureCollectionInput] = None,
    anchor_plate_id: int = None,
    target_extent: Optional[_PathLike] = None,
    output_dir: _PathLike = os.curdir,
    buffer_distance: float = 6,
    clip_to_overriding_plate: bool = False,
    return_output: bool = False,
) -> Optional[gpd.GeoDataFrame]:

    if plate_reconstruction is None:
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)
        if not isinstance(topology_features, pygplates.FeatureCollection):
            topology_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(topology_features).get_features()
                )
        if not isinstance(static_polygons, pygplates.FeatureCollection):
            static_polygons = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(static_polygons).get_features()
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
            )
    else:
        rotation_model = plate_reconstruction.rotation_model
        topology_features = plate_reconstruction.topology_features
        static_polygons = plate_reconstruction.static_polygons

    gplot = PlotTopologies(plate_reconstruction, anchor_plate_id=anchor_plate_id)
    gplot.time = float(time)
    plate_polygons = gplot.get_all_topologies()
    plate_polygons["feature_type"] = plate_polygons["feature_type"].astype(str)
    plate_polygons = plate_polygons[
        plate_polygons["feature_type"].isin({
            "gpml:TopologicalClosedPlateBoundary",
            "gpml:OceanicCrust",
            "gpml:TopologicalNetwork",
        })
    ]

    topologies = _extract_overriding_plates(
        time=time,
        topology_features=topology_features,
        rotation_model=rotation_model,
        anchor_plate_id=anchor_plate_id,
    )
    plate_polygons.crs = topologies.crs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        topologies = topologies[
            (topologies["over"] != -1)
            & (topologies["over"] != 0)
            & (topologies["polarity"] != "None")
        ]
        topologies = topologies.explode(ignore_index=True)
        topologies = _merge_lines(topologies)
        buffered = {}
        for _, row in topologies.iterrows():
            _buffer_sz(row, buffer_distance, topologies.crs, out=buffered)
        buffered = gpd.GeoDataFrame(buffered, geometry="geometry", crs=topologies.crs)

        if clip_to_overriding_plate:
            clipped = []
            for plate_id in buffered["over"].unique():
                try:
                    poly_match = plate_polygons[
                        plate_polygons["reconstruction_plate_ID"] == plate_id
                    ]
                    if poly_match.empty:
                        print(f"[WARN] No plate polygon match for plate_id: {plate_id}")
                        continue
    
                    intersection = gpd.overlay(
                        buffered[buffered["over"] == plate_id],
                        poly_match,
                    )
    
                    if not intersection.empty:
                        clipped.append(intersection)
    
                except Exception as e:
                    print(f"[ERROR] Clipping failed for plate_id {plate_id}: {e}")
    
            if clipped:
                clipped = gpd.GeoDataFrame(pd.concat(clipped, ignore_index=True))
                clipped = clipped[["name", "polarity", "feature_type", "over", "geometry"]]
                clipped = clipped.rename(columns={"over": "plate_id", "feature_type": "ftype"})
                buffered = gpd.GeoDataFrame(clipped, geometry="geometry")

    if not buffered.geometry.is_valid.all():
        buffered.geometry = buffered.buffer(0)
        
    # Clip to target extent shapefile
    if target_extent is not None:
        extent_gdf = gpd.read_file(target_extent)
        if extent_gdf.crs != buffered.crs:
            extent_gdf = extent_gdf.to_crs(buffered.crs)
        buffered = gpd.clip(buffered, extent_gdf)
        buffered = buffered[~buffered.geometry.is_empty & buffered.geometry.notna()]

    if output_dir is not None:
        output_filename = os.path.join(output_dir, f"buffer_zones_{time:0.0f}Ma.geojson")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            buffered.to_file(output_filename)
            
    if return_output:
        return buffered
    
    return None


def _buffer_sz(row, distance_degrees, crs, out):
    
    geom = gpd.GeoSeries(row["geometry"], crs=crs)
    point = geom.representative_point()
    proj = f"+proj=aeqd +lat_0={point.y.iloc[0]} +lon_0={point.x.iloc[0]} +x_0=0 +y_0=0"
    projected = geom.to_crs(proj)

    distance_metres = np.deg2rad(distance_degrees) * EARTH_RADIUS * 1000.0
    direction = 1.0 if str(row["polarity"]).lower() == "left" else -1.0
    projected_buffered = projected.buffer(distance_metres * direction, single_sided=True, join_style="round") # join_style="bevel"
    projected_buffered = projected_buffered.intersection(projected.buffer(abs(distance_metres)))
    buffered = projected_buffered.to_crs(crs)
    geometry_out = buffered.iloc[0]
    
    # Skip bad geometries
    if not _has_enough_points(geometry_out):
        return out

    # Decompose MultiPolygon
    parts = list(geometry_out.geoms) if isinstance(geometry_out, MultiPolygon) else [geometry_out]
    
    geometries_out = []
    for part in parts:
        try:
            wrapped = wrap_geometries(part, central_meridian=0.0, tessellate_degrees=0.1)
            if isinstance(wrapped, (list, tuple)):
                geometries_out.extend(wrapped)
            elif wrapped is not None:
                geometries_out.append(wrapped)
        except Exception as e:
            print(f"[WARN] Failed to wrap geometry: {e}")
            continue

    if isinstance(geometries_out, BaseGeometry):
        geometries_out = [geometries_out]

    # Append results
    for i in geometries_out:
        for column_name in row.index:
            if column_name == "geometry":
                continue
            out.setdefault(column_name, []).append(row[column_name])
        out.setdefault("geometry", []).append(i)

    return out


def _has_enough_points(geometry, min_points=3):
    if geometry is None or geometry.is_empty:
        return False
    if isinstance(geometry, Polygon):
        return len(geometry.exterior.coords) >= min_points
    elif isinstance(geometry, MultiPolygon):
        return any(len(poly.exterior.coords) >= min_points for poly in geometry.geoms)
    return False


def _extract_overriding_plates(
    time,
    topology_features,
    rotation_model,
    anchor_plate_id: int = None,
):
    
    resolved_sections = []
    pygplates.resolve_topologies(
        topological_features=topology_features,
        rotation_model=rotation_model,
        resolved_topologies=[],  # Discard boundaries/networks
        reconstruction_time=float(time),
        resolved_topological_sections=resolved_sections,
        anchor_plate_id=anchor_plate_id,
    )

    # Ignore flat slab topologies
    slab_types = {
        pygplates.FeatureType.gpml_slab_edge,
        pygplates.FeatureType.gpml_topological_slab_boundary,
    }
    resolved_sections = [
        i
        for i in resolved_sections
        if i.get_topological_section_feature().get_feature_type()
        not in slab_types
    ]

    geometries = []
    polarities = []
    names = []
    feature_types = []
    feature_ids = []
    plate_ids = []
    overriding_plates = []
    subducting_plates = []
    left_plates = []
    right_plates = []
    shared_1s = []
    shared_2s = []
    for i in resolved_sections:
        for segment in i.get_shared_sub_segments():
            geometry = segment.get_resolved_geometry()
            geometry = pygplates_to_shapely(geometry, tessellate_degrees=0.1)

            polarity = segment.get_feature().get_enumeration(
                pygplates.PropertyName.gpml_subduction_polarity,
                "None",
            )
            if polarity == "Unknown":
                polarity = "None"
            valid_polarities = {"None", "Left", "Right"}
            if polarity not in valid_polarities:
                warnings.warn(
                    "Unknown polarity: {}".format(polarity), RuntimeWarning
                )
                continue

            name = segment.get_feature().get_name()
            if "flat slab" in name.lower():
                continue

            feature_type = (
                segment.get_feature().get_feature_type().to_qualified_string()
            )
            feature_id = segment.get_feature().get_feature_id().get_string()
            plate_id = segment.get_feature().get_reconstruction_plate_id(-1)
            tmp = segment.get_overriding_and_subducting_plates()
            if tmp is None:
                overriding_plate = -1
                subducting_plate = -1
            else:
                overriding_plate, subducting_plate = tmp
                overriding_plate = (
                    overriding_plate.get_feature().get_reconstruction_plate_id(
                        -1
                    )
                )
                subducting_plate = (
                    subducting_plate.get_feature().get_reconstruction_plate_id(
                        -1
                    )
                )
            del tmp
            left_plate = segment.get_feature().get_left_plate(-1)
            right_plate = segment.get_feature().get_right_plate(-1)

            sharing_topologies = segment.get_sharing_resolved_topologies()
            if len(sharing_topologies) > 0:
                shared_1 = (
                    sharing_topologies[0]
                    .get_feature()
                    .get_reconstruction_plate_id(-1)
                )
            else:
                shared_1 = -1
            if len(sharing_topologies) > 1:
                shared_2 = (
                    sharing_topologies[1]
                    .get_feature()
                    .get_reconstruction_plate_id(-1)
                )
            else:
                shared_2 = -1

            geometries.append(geometry)
            polarities.append(polarity)
            names.append(name)
            feature_types.append(feature_type)
            feature_ids.append(feature_id)
            plate_ids.append(plate_id)
            overriding_plates.append(overriding_plate)
            subducting_plates.append(subducting_plate)
            left_plates.append(left_plate)
            right_plates.append(right_plate)
            shared_1s.append(shared_1)
            shared_2s.append(shared_2)

    gdf = gpd.GeoDataFrame(
        {
            "polarity": polarities,
            "geometry": geometries,
            "name": names,
            "type": feature_types,
            "id": feature_ids,
            "plate_id": plate_ids,
            "over": overriding_plates,
            "subd": subducting_plates,
            "left": left_plates,
            "right": right_plates,
            "shared_1": shared_1s,
            "shared_2": shared_2s,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    
    return gdf


def _merge_lines(
    data: gpd.GeoDataFrame,
    groupby: Iterable[Hashable] = ("polarity", "type", "over"),
):
    
    out = []
    for gb_vals, grouped in data.groupby(list(groupby)):
        geom = linemerge(grouped.geometry.to_list(), directed=True)
        if isinstance(geom, BaseMultipartGeometry):
            geom = list(geom.geoms)
        else:
            geom = [geom]
        gb_data = {
            "geometry": geom,
            **{
                gb_col: gb_val
                for gb_col, gb_val
                in zip(groupby, gb_vals)
            }
        }
        if "name" not in gb_data.keys():
            gb_data["name"] = ":".join(grouped["name"].unique())
        out.append(
            gpd.GeoDataFrame(gb_data, geometry="geometry")
        )
    out = gpd.GeoDataFrame(
        pd.concat(out, ignore_index=True),
        geometry="geometry",
        crs=data.crs,
    )
    
    return out


def prepare_deposit_data(
    deposit_data,
    buffer_zones_dir,
    output_filename,
    time_steps,
    plate_reconstruction=None,
    rotation_model=None,
    topology_features=None,
    static_polygons=None,
    anchor_plate_id: int = None,
    min_time=-np.inf,
    max_time=np.inf,
    n_jobs=-2,
    verbose=False,
):
    
    """
    Prepare and reconstruct deposit data for analysis within buffer zones.

    This function loads and processes deposit occurrence data, partitions deposits
    into plates, reconstructs their paleopositions, filters them against precomputed
    buffer zones, and assigns overriding plate IDs at each time step. It enables
    integration of deposit data into plate tectonic reconstructions.

    Parameters
    ----------
    deposit_data : str, pd.DataFrame, or array-like
        Deposit dataset containing at least columns ["lat", "lon", "age (Ma)"].
        Can be provided as a CSV filename or a pandas DataFrame.
    buffer_zones_dir : str or Path-like
        Directory containing buffer zone polygons for each reconstruction time
        (GeoJSON or Shapefile), typically generated by `run_create_buffer_zones`.
    output_filename : str or Path-like
        Output CSV filename to save processed results. If None, no file is written.
    time_steps : sequence of float
        Geological time steps (in Ma) to which deposits should be snapped and
        reconstructed.
    plate_reconstruction : PlateReconstruction, optional
        Pre-initialised reconstruction object. If not provided, `rotation_model`,
        `topology_features`, and `static_polygons` must be specified.
    rotation_model : str, Path, pygplates.RotationModel, or FeatureCollection, optional
        Rotation model for plate motions if no PlateReconstruction is supplied.
    topology_features : str, Path, FeatureCollection, or sequence, optional
        Topological plate boundary features used to resolve dynamic boundaries.
    static_polygons : str, Path, FeatureCollection, or sequence, optional
        Present-day static polygons used for partitioning deposits into plates.
    anchor_plate_id : int
        Anchor plate ID for reconstruction.
    min_time, max_time : float, optional
        Age filter for deposits (in Ma). Deposits outside this range are discarded.
    n_jobs : int, default=-2
        Number of parallel processes for cleaning and plate assignment. Follows joblib
        convention (-1 for all CPUs, -2 for all but one).
    verbose : bool, default=False
        Print progress and diagnostic messages if True.

    Returns
    -------
    pd.DataFrame
        Processed deposit data with reconstructed paleocoordinates and plate IDs.
        Contains columns such as:
        - present_lat, present_lon : Original coordinates
        - age (Ma) : Snapped to nearest reconstruction time
        - lon, lat : Reconstructed paleocoordinates
        - plate_id : Plate ID assigned during partitioning
        - overriding_plate_id : Plate ID of overriding plate from topologies

    Notes
    -----
    - Deposits are first snapped to the nearest `time_steps` before reconstruction.
    - Points outside buffer zones are excluded.
    - Output is chronologically sorted and written to file if `output_filename` is provided.
    - Buffer zone polygons must already exist in `buffer_zones_dir`.
    """
    
    # Build/unwrap reconstruction context
    if plate_reconstruction is None:
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)
        if not isinstance(topology_features, pygplates.FeatureCollection):
            topology_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(topology_features).get_features()
            )
        if not isinstance(static_polygons, pygplates.FeatureCollection):
            static_polygons = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(static_polygons).get_features()
            )
        plate_reconstruction = PlateReconstruction(
            rotation_model=rotation_model,
            topology_features=topology_features,
            static_polygons=static_polygons,
            anchor_plate_id=anchor_plate_id,
        )

    # Load and prefilter
    if isinstance(deposit_data, str):
        if verbose:
            print("Loading deposit data from: " + deposit_data, file=stderr)
        deposit_data = pd.read_csv(deposit_data)
    else:
        deposit_data = pd.DataFrame(deposit_data)

    deposit_data = deposit_data.drop(columns=["index"], errors="ignore")

    deposit_data["age (Ma)"] = pd.to_numeric(deposit_data["age (Ma)"], errors="coerce")
    deposit_data = deposit_data[
        (deposit_data["age (Ma)"] >= min_time) & (deposit_data["age (Ma)"] <= max_time)
    ]

    # Ensure weight exists (if not provided, use 1)
    if "weight" not in deposit_data.columns:
        deposit_data["weight"] = 1

    # Reconstruct and clean
    deposit_data = _partition_and_reconstruct(
        deposit_data=deposit_data,
        plate_reconstruction=plate_reconstruction,
        anchor_plate_id=anchor_plate_id,
        time_steps=time_steps,
    )
    deposit_data = _clean_deposit_data(
        deposit_data=deposit_data,
        polygons_dir=buffer_zones_dir,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Finalize columns and write
    deposit_data = deposit_data.sort_values(by="age (Ma)").reset_index(drop=True)

    cols = ["present_lat", "present_lon", "age (Ma)", "weight", "lon", "lat"]
    deposit_data = deposit_data[cols]

    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.exists(output_dir):
            if verbose:
                print("Output directory does not exist; creating now: " + output_dir, file=stderr)
            os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print("Writing output to file: " + os.path.basename(output_filename), file=stderr)
        deposit_data.to_csv(output_filename, index=False)

    return deposit_data


def _partition_and_reconstruct(deposit_data, plate_reconstruction, anchor_plate_id, time_steps):
    
    time_steps = np.array(time_steps)

    # Snap ages to nearest timestep
    deposit_data["age (Ma)"] = deposit_data["age (Ma)"].astype(float).apply(
        lambda x: time_steps[np.abs(time_steps - x).argmin()]
    )

    rotation_model = plate_reconstruction.rotation_model
    static_polygons = plate_reconstruction.static_polygons

    # Build temporary features at present-day coords
    features = []
    for index, row in deposit_data.iterrows():
        lon = float(row["lon"])
        lat = float(row["lat"])
        name = str(index)
        age = float(row["age (Ma)"])

        geom = pygplates.PointOnSphere(lat, lon)
        feature = pygplates.Feature()
        feature.set_geometry(geom)
        feature.set_valid_time(age, 0.0)
        feature.set_name(name)
        features.append(feature)

    # Rename original coords and prepare reconstructed columns
    deposit_data = deposit_data.rename(columns={"lon": "present_lon", "lat": "present_lat"})
    deposit_data["lon"] = np.nan
    deposit_data["lat"] = np.nan

    # Partition and reconstruct (no need to save plate_id to dataframe)
    partitioned = pygplates.partition_into_plates(
        partitioning_features=static_polygons,
        rotation_model=rotation_model,
        features_to_partition=features,
    )

    reconstructed = []
    times = set(f.get_valid_time()[0] for f in partitioned)
    for time in times:
        to_reconstruct = [f for f in partitioned if f.get_valid_time()[0] == time]
        pygplates.reconstruct(to_reconstruct, rotation_model, reconstructed, time, anchor_plate_id)

    for f in reconstructed:
        geom = f.get_reconstructed_geometry()
        feature = f.get_feature()
        lat, lon = geom.to_lat_lon()
        index = int(feature.get_name())
        deposit_data.at[index, "lon"] = lon
        deposit_data.at[index, "lat"] = lat

    return deposit_data


def _clean_deposit_data(deposit_data, polygons_dir, n_jobs, verbose=False):
    
    times = deposit_data["age (Ma)"].unique()

    with Parallel(n_jobs, verbose=int(verbose)) as p:
        out = p(
            delayed(_clean_timestep)(
                (deposit_data[deposit_data["age (Ma)"] == time]).copy(),
                polygons_dir,
                time,
            )
            for time in times
        )
        
    return pd.concat(out, ignore_index=True)


def _clean_timestep(deposit_data, polygons_dir, time):
    
    polygons_filename = os.path.join(
        polygons_dir, f"buffer_zones_{time:0.0f}Ma.geojson"
    )
    if not os.path.isfile(polygons_filename):
        polygons_filename = os.path.join(
            polygons_dir, f"buffer_zones_{time:0.0f}Ma.shp"
        )
    polygons = gpd.read_file(polygons_filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        union = polygons.unary_union
    valid = []
    for _, row in deposit_data.iterrows():
        p = Point(row["lon"], row["lat"])
        if union.contains(p):
            valid.append(True)
        else:
            valid.append(False)

    return deposit_data[valid]


def partition_and_reconstruct(
        deposit_data,
        plate_reconstruction,
        time_steps,
        output_filename,
        anchor_plate_id=None,
        verbose=False):
    
    """
    Partition deposit occurrences into plates and reconstruct their paleocoordinates
    through geological time.

    This function takes a dataset of deposit locations with ages, assigns each deposit
    to a tectonic plate using static polygons, and reconstructs their positions at
    specified geological time steps. The results include present-day coordinates,
    assigned plate IDs, and reconstructed coordinates for each requested time step.

    Parameters
    ----------
    deposit_data : str, pd.DataFrame, or array-like
        Deposit dataset containing at least columns ["lon", "lat", "age (Ma)"].
        Can be a CSV filename, a pandas DataFrame, or a similar array-like object.
    plate_reconstruction : PlateReconstruction
        Pre-initialised reconstruction object containing the rotation model and
        static polygons needed for partitioning and reconstruction.
    time_steps : sequence of float
        Geological time steps (in Ma) at which deposits should be reconstructed.
        Deposit ages are snapped to the nearest time step.
    output_filename : str or Path-like
        Path to CSV file where the processed dataset will be written. If None,
        no file is written.
    anchor_plate_id : int
        Anchor plate ID for reconstruction.
    verbose : bool, default=False
        If True, print progress and warnings during processing.

    Returns
    -------
    pd.DataFrame
        Processed deposit data with columns including:
        - present_lon, present_lat : Original coordinates
        - age (Ma) : Snapped to nearest reconstruction time
        - plate_id : Plate ID assigned during partitioning
        - lon_{t}, lat_{t} : Reconstructed coordinates at each time step t

    Notes
    -----
    - Deposits are assigned to plates at present day using `pygplates.partition_into_plates`.
    - Each deposit is then reconstructed back in time using `pygplates.reconstruct`.
    - Deposits are only reconstructed at times younger than their formation age.
    - Deposits that cannot be assigned to a plate are skipped and flagged with NaN values.
    - The output is sorted by age and can be saved to file.
    """
    
    rotation_model = plate_reconstruction.rotation_model
    static_polygons = plate_reconstruction.static_polygons
    
    # Load data if filename provided
    if isinstance(deposit_data, str):
        if verbose:
            print(f"Loading deposit data from {deposit_data}")
        deposit_data = pd.read_csv(deposit_data)
    
    deposit_data["age (Ma)"] = pd.to_numeric(deposit_data["age (Ma)"], errors="coerce")
    deposit_data = deposit_data[
        (deposit_data["age (Ma)"] >= min(time_steps))
        & (deposit_data["age (Ma)"] <= max(time_steps))
    ]
    
    time_steps = np.array(time_steps)
    
    deposit_data["age (Ma)"] = deposit_data["age (Ma)"].astype(float).apply(
        lambda x: time_steps[np.abs(time_steps - x).argmin()]
        )
    
    # Rename original coordinates if not already done
    if "present_lon" not in deposit_data.columns:
        deposit_data = deposit_data.rename(
            columns={
                "lon": "present_lon",
                "lat": "present_lat",
            }
        )
    
    # Add plate_id column
    deposit_data["plate_id"] = np.nan
    
    # Create columns for each time step
    new_cols = {f"lon_{time}": np.nan for time in time_steps}
    new_cols.update({f"lat_{time}": np.nan for time in time_steps})
    deposit_data = pd.concat([deposit_data, pd.DataFrame(new_cols, index=deposit_data.index)], axis=1)
    
    # Process each deposit
    deposit_data = deposit_data.reset_index(drop=True)
    for index, row in deposit_data.iterrows():
        if verbose and index % 100 == 0:
            print(f"Reconstructing deposit {index}/{len(deposit_data)}")
            
        try:
            lon = float(row["present_lon"])
            lat = float(row["present_lat"])
            age = float(row["age (Ma)"])
            name = str(index)
            
            # Create point feature for the deposit
            geom = pygplates.PointOnSphere(lat, lon)
            feature = pygplates.Feature()
            feature.set_geometry(geom)
            
            # Instead of setting valid time with formation age,
            # we will handle time filtering during reconstruction
            # Just set a very large valid time range to avoid errors
            feature.set_name(name)
            
            # Partition this single feature without time constraints
            partitioned_features = pygplates.partition_into_plates(
                partitioning_features=static_polygons,
                rotation_model=rotation_model,
                features_to_partition=[feature],
            )
            
            # Skip if could not assign to a plate
            if not partitioned_features:
                if verbose:
                    print(f"  Deposit {index} could not be assigned to a plate")
                continue
                
            # Get plate ID
            plate_id = int(partitioned_features[0].get_reconstruction_plate_id())
            deposit_data.at[index, "plate_id"] = plate_id
            
            # Reconstruct for each time step
            for time in time_steps:
                # Skip times before formation
                if time > age:
                    continue
                    
                reconstructed = []
                try:
                    pygplates.reconstruct(
                        partitioned_features,
                        rotation_model,
                        reconstructed,
                        time,
                        anchor_plate_id,
                    )
                    
                    # If reconstruction successful, update coordinates
                    if reconstructed:
                        geom = reconstructed[0].get_reconstructed_geometry()
                        lat_rec, lon_rec = geom.to_lat_lon()
                        deposit_data.at[index, f"lon_{time}"] = lon_rec
                        deposit_data.at[index, f"lat_{time}"] = lat_rec
                except Exception as e:
                    if verbose:
                        print(f"  Error reconstructing deposit {index} at time {time}: {e}")
        except Exception as e:
            if verbose:
                print(f"  Error processing deposit {index}: {e}")
    
    if verbose:
        print("Reconstruction complete")
        
    deposit_data = deposit_data.sort_values(by="age (Ma)").reset_index(drop=True)
        
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
        deposit_data.to_csv(output_filename, index=False)
        
    return deposit_data


def generate_unlabelled_points(
    times,
    input_dir,
    num,
    output_filename=None,
    seed=None,
    rotation_model=None,
    topology_features=None,
    static_polygons=None,
    anchor_plate_id=None,
    n_jobs=1,
    verbose=False,
):
    
    """
    Generate uniformly distributed random points within buffer zones across 
    geological time steps, and reconstruct them to present-day coordinates.

    This function samples random geographic points constrained to predefined
    buffer zone polygons, then reconstructs their paleocoordinates to the
    present day using a plate motion model. It supports parallel execution for
    multiple time steps.

    Parameters
    ----------
    times : sequence of float
        Geological times (in Ma) at which to generate points.
    input_dir : str or Path-like
        Directory containing buffer zone files (`buffer_zones_{time}Ma.geojson` or `.shp`).
    num : int
        Number of points to generate per time step.
    output_filename : str or Path-like, optional
        Output CSV file to save generated points. If None, results are not written.
    seed : int, optional
        Random seed for reproducibility of point generation.
    rotation_model : str, Path, pygplates.RotationModel, or FeatureCollection, optional
        Rotation model used for reconstruction.
    topology_features : str, Path, FeatureCollection, or sequence, optional
        Topological features required for reconstruction.
    static_polygons : str, Path, FeatureCollection, or sequence, optional
        Static plate polygons used to assign plate IDs during reconstruction.
    anchor_plate_id : int
        Anchor plate ID for reconstruction.
    n_jobs : int, default=1
        Number of parallel threads used for point generation.
    verbose : bool, default=False
        If True, print progress messages during execution.

    Returns
    -------
    pd.DataFrame
        DataFrame of generated points with columns:
        - lon, lat : Paleocoordinates at the specified geological time
        - present_lon, present_lat : Reconstructed coordinates at present day
        - age (Ma) : Geological age of reconstruction
        - weight : Column set to 1 for all rows (useful for later weighting)

    Notes
    -----
    - Points are generated uniformly on the sphere, but only those within 
      the buffer zone polygons are retained.
    - If buffer zone polygons are missing or invalid, no points are generated
      for that time step.
    - Reverse reconstruction is performed so that generated paleopoints 
      can be mapped to present-day coordinates.
    - Output is sorted by geological age before being returned.
    """

    seq = np.random.SeedSequence(entropy=seed)
    rngs = [np.random.default_rng(i) for i in seq.spawn(n_jobs)]
    times_split = np.array_split(times, n_jobs)

    with Parallel(n_jobs, verbose=int(verbose)) as p:
        results = p(
            delayed(_multiple_timesteps_unlabelled)(
                times=t,
                input_dir=input_dir,
                num=num,
                rng=rng,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
            )
            for t, rng in zip(times_split, rngs)
        )
    results_flattened = []
    for i in results:
        results_flattened.extend(i)
    results = results_flattened
    del results_flattened

    results = pd.concat(results, ignore_index=True).sort_values(by="age (Ma)")
    results["weight"] = 1
    results = results[["present_lat", "present_lon", "age (Ma)", "weight", "lon", "lat"]]

    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.isdir(output_dir):
            if verbose:
                print(
                    "Output directory does not exist; creating now: "
                    + output_dir,
                    file=stderr,
                )
            os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(
                "Writing output to file: " + str(output_filename),
                file=stderr,
            )
        results.to_csv(output_filename, index=False)
        
    return results


def _multiple_timesteps_unlabelled(
        times,
        input_dir,
        num,
        rng,
        rotation_model,
        topology_features,
        static_polygons,
        anchor_plate_id,
):
    
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(seed=rng)

    if not isinstance(rotation_model, pygplates.RotationModel):
        rotation_model = pygplates.RotationModel(rotation_model)
    if not isinstance(topology_features, pygplates.FeatureCollection):
        topology_features = pygplates.FeatureCollection(
            pygplates.FeaturesFunctionArgument(topology_features).get_features()
        )
    if not isinstance(static_polygons, pygplates.FeatureCollection):
        static_polygons = pygplates.FeatureCollection(
            pygplates.FeaturesFunctionArgument(static_polygons).get_features()
            )

    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
        anchor_plate_id=anchor_plate_id,
    )

    out = []
    for time in times:
        out.append(
            _generate_points_timestep(
                time=time,
                input_dir=input_dir,
                plate_reconstruction=plate_reconstruction,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
                num=num,
                rng=rng,
            )
        )
        
    return out


def _generate_points_timestep(
    time,
    input_dir,
    plate_reconstruction,
    rotation_model,
    topology_features,
    static_polygons,
    anchor_plate_id,
    num,
    rng,
):
    
    input_filename = os.path.join(
        input_dir, f"buffer_zones_{time:0.0f}Ma.geojson"
    )
    if not os.path.isfile(input_filename):
        input_filename = os.path.join(
            input_dir, f"buffer_zones_{time:0.0f}Ma.shp"
        )
    gdf = gpd.read_file(input_filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # No valid buffer zone, return empty DataFrame
        if gdf.area.sum() <= 0.0:
            return pd.DataFrame(
                columns=[
                    "lon",
                    "lat",
                    "present_lon",
                    "present_lat",
                    "age (Ma)",
                ],
            )

    points = np.full((num, 2), np.nan)
    to_fill = np.where(np.any(np.isnan(points), axis=1))[0]
    num_to_fill = to_fill.size
    while num_to_fill > 0:
        generated_points = _generate_points(
            n=num_to_fill,
            output_format="degrees",
            order="lonlat",
            n_jobs=1,
            rng=rng,
        )
        generated_points[
            ~_points_in_polygons(generated_points, gdf["geometry"])
        ] = np.nan
        points[to_fill] = generated_points

        to_fill = np.where(np.any(np.isnan(points), axis=1))[0]
        num_to_fill = to_fill.size

    if (
        plate_reconstruction is not None
        or (topology_features is not None and rotation_model is not None)
    ):
        present_day_coords = _reconstruct_points_present(
            data=pd.DataFrame({"lon": points[:, 0], "lat": points[:, 1], "age (Ma)": time}),
            plate_reconstruction=plate_reconstruction,
            rotation_model=rotation_model,
            static_polygons=static_polygons,
            anchor_plate_id=anchor_plate_id,
            )
    else:
        present_day_coords = pd.DataFrame(
            {
                "lon_0": np.full_like(points, np.nan),
                "lat_0": np.full_like(points, np.nan),
            }
        )

    try:
        out = pd.DataFrame(
            {
                "lon": points[:, 0],
                "lat": points[:, 1],
                "present_lon": present_day_coords["lon_0"],
                "present_lat": present_day_coords["lat_0"],
                "age (Ma)": time,
            }
        )
    except IndexError as err:
        print(present_day_coords)
        raise err
        
    return out


def _generate_points(
    n=1, output_format="radians", order="lonlat", n_jobs=1, rng=None
):

    valid_output_formats = {
        "radians",
        "degrees",
        "xyz",
    }
    valid_orders = {
        "lonlat",
        "latlon",
    }

    seed = 42

    output_format = str(output_format).lower()
    if output_format not in valid_output_formats:
        raise ValueError("Invalid `output_format`: " + output_format)
    order = str(order).lower()
    if order not in valid_orders:
        raise ValueError("Invalid `order`: " + order)

    if n_jobs == 1:
        if rng is None:
            rng = np.random.default_rng(seed=seed)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Invalid `rng` type: " + str(type(rng)))
        xyz = _generate_points_(n=n, rng=rng)
    else:
        if rng is None:
            rng = np.random.SeedSequence(seed)
        if not isinstance(rng, np.random.SeedSequence):
            raise TypeError("Invalid `rng` type: " + str(type(rng)))
        xyz = _generate_points_parallel(n=n, n_jobs=n_jobs, seq=rng)

    if output_format == "xyz":
        return xyz
    lon, lat = xyz2lonlat(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], degrees=False)
    lon = np.array(lon)
    lat = np.array(lat)
    out = np.hstack((lon.reshape((-1, 1)), lat.reshape((-1, 1))))
    if order == "latlon":
        out = np.fliplr(out)
    if output_format == "degrees":
        out = np.rad2deg(out)
        
    return out


def _generate_points_(n=1, rng=None):
    
    seed = 42

    if rng is None:
        rng = np.random.default_rng(seed=seed)
    xyz = np.zeros((n, 3))
    zero_rows = np.where(np.all(np.isclose(xyz, 0.0), axis=1))[0]
    num_rows = zero_rows.size
    while num_rows > 0:
        tmp = rng.standard_normal(size=(num_rows, 3))
        xyz[zero_rows] = tmp
        zero_rows = np.where(np.all(np.isclose(xyz, 0.0), axis=1))[0]
        num_rows = zero_rows.size

    xyz /= np.sqrt((xyz ** 2).sum(axis=1)).reshape((-1, 1))
    
    return xyz


def _generate_points_parallel(n=1, n_jobs=2, seq=None):
    
    seed = 42

    if seq is None:
        seq = np.random.SeedSequence(seed)
    generators = [np.random.default_rng(i) for i in seq.spawn(n_jobs)]
    xyz = np.zeros((n, 3))

    executor = concurrent.futures.ThreadPoolExecutor(n_jobs)
    step = np.ceil(n / n_jobs).astype(np.int_)

    def _fill(random_state, out, first, last):
        zero_rows = np.where(np.all(np.isclose(out[first:last], 0.0), axis=1))[
            0
        ]
        num_rows = zero_rows.size
        while num_rows > 0:
            random_state.standard_normal(
                size=(num_rows, 3), out=out[first:last]
            )
            zero_rows = np.where(
                np.all(np.isclose(out[first:last], 0.0), axis=1)
            )[0]
            num_rows = zero_rows.size
        out[first:last] /= np.sqrt((out[first:last] ** 2).sum(axis=1)).reshape(
            (-1, 1)
        )

    futures = {}
    for i in range(n_jobs):
        args = (_fill, generators[i], xyz, i * step, (i + 1) * step)
        futures[executor.submit(*args)] = i
    concurrent.futures.wait(futures)

    executor.shutdown(False)
    
    return xyz


def _points_in_polygons(points, polygons):
    
    polygons_sorted = sorted(polygons, key=lambda x: x.area, reverse=True)
    out = np.zeros(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        p = Point(points[i, 0], points[i, 1])
        for polygon in polygons_sorted:
            if polygon.contains(p):
                out[i] = True
                break
        else:
            out[i] = False
            
    return out


def _reconstruct_points_present(
    data,
    plate_reconstruction=None,
    rotation_model=None,
    static_polygons=None,
    anchor_plate_id=None,
):
    
    if plate_reconstruction is not None:
        rotation_model = plate_reconstruction.rotation_model
        static_polygons = plate_reconstruction.static_polygons

    if rotation_model is None:
        raise TypeError("Rotation model must be provided")

    data = _load_data(data, copy=False)

    if not isinstance(rotation_model, pygplates.RotationModel):
        rotation_model = pygplates.RotationModel(rotation_model)

    # Get the reconstruction time from the data
    reconstruction_time = float(data["age (Ma)"].iloc[0])

    # Prepare points
    lons = data["lon"].values
    lats = data["lat"].values
    # points = pygplates.MultiPointOnSphere(np.column_stack((lats, lons)))

    # Reconstruct once from reconstruction_time to 0 Ma
    reconstructed_points = reverse_reconstruct_points_impl(
        lons=lons,
        lats=lats,
        rotation_model=rotation_model,
        static_polygons=static_polygons,
        time=reconstruction_time,
        anchor_plate_id= anchor_plate_id,
    )

    new_lats, new_lons = reconstructed_points[0]['lats'], reconstructed_points[0]['lons']
    data["lon_0"] = new_lons
    data["lat_0"] = new_lats

    return data


def _load_data(
    data: _PathOrDataFrame,
    verbose: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    
    if not isinstance(data, pd.DataFrame):
        if verbose:
            print(f"Loading data from file: {data}", file=stderr)
        data = pd.read_csv(data)
    elif copy:
        data = pd.DataFrame(data)
        
    return data


def generate_grid_points(
    times,
    resolution,
    polygons_dir,
    rotation_model=None,
    topology_features=None,
    static_polygons=None,
    anchor_plate_id=None,
    output_filename=None,
    n_jobs=1,
    verbose=False,
):
    
    """
    Generate a regular grid of points within buffer zones across geological times.

    This function creates a latitude–longitude grid at a specified resolution,
    clips it to predefined buffer zone polygons, and reconstructs the resulting points
    to present-day coordinates using a plate motion model. It supports parallelisation
    across time steps.

    Parameters
    ----------
    times : sequence of float
        Geological times (in Ma) at which grid points will be generated.
    resolution : float
        Grid spacing in degrees for both latitude and longitude.
    polygons_dir : str or Path-like
        Directory containing buffer zone files named
        `buffer_zones_{time}Ma.geojson` or `.shp`.
    rotation_model : str, Path, pygplates.RotationModel, or FeatureCollection, optional
        Rotation model for plate motions, required for reconstruction if
        `plate_reconstruction` is not provided.
    topology_features : str, Path, FeatureCollection, or sequence, optional
        Topological features used to resolve dynamic plate boundaries.
    static_polygons : str, Path, FeatureCollection, or sequence, optional
        Present-day static polygons used for partitioning points into plates.
    output_filename : str or Path-like, optional
        Output CSV file to save processed points. If None, results are not written.
    n_jobs : int, default=1
        Number of parallel jobs.
    verbose : bool, default=False
        Print progress and diagnostic information.

    Returns
    -------
    pd.DataFrame
        DataFrame containing grid points with columns:
        - lon, lat : Paleocoordinates at the specified time
        - present_lon, present_lat : Reconstructed present-day coordinates
        - age (Ma) : Geological time associated with the grid points

    Notes
    -----
    - Points are generated on a global grid and clipped to buffer zones for each time step.
    - If no intersection is found between grid and buffer zones, an empty DataFrame
      is returned for that time.
    - Reverse reconstruction is used so that past-time points can be mapped to
      present-day positions.
    - Output is concatenated across all time steps and optionally saved to file.
    """
    
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    elif n_jobs < 0:
        n_jobs = cpu_count() + n_jobs + 1

    times = np.array(times)
    times_split = [times[i::n_jobs] for i in range(n_jobs)]

    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        out = parallel(
            delayed(_grid_points_subset)(
                times=t,
                resolution=resolution,
                polygons_dir=polygons_dir,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
                verbose=verbose,
            )
            for t in times_split
        )
    out = pd.concat(out, ignore_index=True)
    out = out.drop(columns="index", errors="ignore")

    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.exists(output_dir):
            if verbose:
                print("Output directory does not exist; creating now: " + output_dir, file=stderr)
            os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print("Writing output to file: " + os.path.basename(output_filename), file=stderr)
        out.to_csv(output_filename, index=False)

    return out


def _grid_points_subset(
    times,
    resolution,
    polygons_dir,
    rotation_model,
    topology_features,
    static_polygons,
    anchor_plate_id,
    verbose=False,
):
    
    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
        anchor_plate_id=anchor_plate_id,
    )

    out = [
        _grid_points_time(
            time=t,
            resolution=resolution,
            polygons_dir=polygons_dir,
            plate_reconstruction=plate_reconstruction,
            anchor_plate_id=anchor_plate_id,
        )
        for t in times
    ]
    return pd.concat(out, ignore_index=True)


def _grid_points_time(
    time,
    resolution,
    polygons_dir,
    plate_reconstruction,
    anchor_plate_id,
):
    
    polygons_filename = os.path.join(
        polygons_dir, f"buffer_zones_{time:0.0f}Ma.geojson"
    )
    if not os.path.isfile(polygons_filename):
        polygons_filename = os.path.join(
            polygons_dir, f"buffer_zones_{time:0.0f}Ma.shp"
        )
    gdf = gpd.read_file(polygons_filename)
    polygons = gdf.geometry

    lons = np.arange(-180, 180 + resolution, resolution)
    lats = np.arange(-90, 90 + resolution, resolution)

    mlons, mlats = np.meshgrid(lons, lats)
    mlons = mlons.reshape((-1, 1))
    mlats = mlats.reshape((-1, 1))
    coords = np.column_stack((mlons, mlats))
    mp = MultiPoint(coords)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        intersection = polygons.unary_union.intersection(mp)
    if hasattr(intersection, "geoms"):
        if len(intersection.geoms) <= 0:
            return pd.DataFrame(
                columns=[
                    "lon",
                    "lat",
                    "present_lon",
                    "present_lat",
                    "age (Ma)",
                ]
            )
        intersection_coords = np.row_stack([i.coords for i in intersection.geoms])
    elif isinstance(intersection, BaseGeometry):
        intersection_coords = np.reshape(intersection.coords, (-1, 2))
    else:
        return pd.DataFrame(
            columns=[
                "lon",
                "lat",
                "present_lon",
                "present_lat",
                "age (Ma)",
            ]
        )
    plons = intersection_coords[:, 0]
    plats = intersection_coords[:, 1]

    if time == 0.0:
        present_lons = np.array(plons)
        present_lats = np.array(plats)
    elif plate_reconstruction is None:
        present_lats = np.full_like(plats, np.nan)
        present_lons = np.full_like(plons, np.nan)
    else:
        present_day_coords = _reconstruct_points_present(
            data=pd.DataFrame(
                {
                    "lon": plons,
                    "lat": plats,
                    "age (Ma)": time,
                }
            ),
            plate_reconstruction=plate_reconstruction,
            anchor_plate_id=anchor_plate_id,
        )
        present_lons = present_day_coords["lon_0"]
        present_lats = present_day_coords["lat_0"]

    out = pd.DataFrame(
        {
            "lon": plons,
            "lat": plats,
            "present_lon": present_lons,
            "present_lat": present_lats,
            "age (Ma)": time,
        }
    )
    
    return out


def run_coregister_point_data(
    point_data: _PathOrDataFrame,
    subduction_data: _PathOrDataFrame,
    output_filename: Optional[_PathLike] = None,
    distance_threshold: Optional[float] = None,
    keep_index=False,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    
    """
    Coregister point data with trench points across geological times.

    This function matches input point data (e.g., deposits or unlabelled samples)
    to the nearest trench point at corresponding geological ages.
    For each point, it appends subduction zone attributes, computes distance to
    the nearest trench, and optionally applies a distance threshold filter.

    Parameters
    ----------
    point_data : str, pd.DataFrame, or array-like
        Input dataset containing columns ["lon", "lat", "age (Ma)"].
        Can be provided as a CSV filename or a pandas DataFrame.
    subduction_data : str, pd.DataFrame, or array-like
        Subduction zone dataset containing reconstructed geometries with
        ["lon", "lat", "age (Ma)"] and relevant attributes. Must include one
        entry per geological time step (matching or rounded from `point_data`).
    output_filename : str or Path-like, optional
        CSV file to save the output dataset. If None, results are not written.
    distance_threshold : float, optional
        Maximum allowed distance (in km) from a trench to consider a point valid.
        If provided, a new column "valid" is added with boolean values.
    keep_index : bool, default=False
        If False, drops the original "index" column from the output.
    n_jobs : int, default=1
        Number of parallel processes. Follows joblib semantics (-1 for all CPUs).
    verbose : bool, default=False
        Print progress and diagnostic information.

    Returns
    -------
    pd.DataFrame
        Coregistered dataset including:
        - All original point columns
        - Subduction zone attributes copied from the nearest trench
        - distance_to_trench (km): great-circle distance to nearest trench
        - valid: (optional) boolean mask if `distance_threshold` is set

    Notes
    -----
    - Distances are computed using the haversine formula and expressed in km.
    - Subduction geometries are matched per time step, using the nearest neighbour
      search on reconstructed coordinates.
    - If `distance_threshold` is not set, all points are retained.
    - Output is sorted by "age (Ma)" (and "label" if present).
    """

    if isinstance(point_data, str):
        if verbose:
            print(
                "Loading point data from file: " + point_data,
                file=stderr,
            )
        point_data = pd.read_csv(point_data)
    else:
        point_data = pd.DataFrame(point_data)

    if isinstance(subduction_data, str):
        if verbose:
            print(
                "Loading subduction data from file: " + subduction_data,
                file=stderr,
            )
        subduction_data = pd.read_csv(subduction_data)
    else:
        subduction_data = pd.DataFrame(subduction_data)

    times = point_data["age (Ma)"].unique()

    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        out = parallel(
            delayed(_coregister_point_data)(
                time=time,
                points=point_data[point_data["age (Ma)"] == time],
                szs=subduction_data[
                    subduction_data["age (Ma)"] == int(np.around(time))
                ],
                distance_threshold=distance_threshold,
            )
            for time in times
        )

    out = pd.DataFrame(pd.concat(out, ignore_index=True))

    if not keep_index:
        out = out.drop(columns="index", errors="ignore")
        
    if "label" in out.columns:
        sort_by = ["label", "age (Ma)"]
    else:
        sort_by = "age (Ma)"
    out = out.sort_values(by=sort_by, ignore_index=True)
    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.isdir(output_dir):
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


def _coregister_point_data(
    time: float,
    points: pd.DataFrame,
    szs: pd.DataFrame,
    distance_threshold: Optional[float] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:

    points = points.copy()
    szs = szs.copy().reset_index()

    # points = points[points["age (Ma)"] == time]
    # szs = szs[szs["age (Ma)"] == int(np.around(time))]

    columns_to_add = set(szs.columns.values) - set(points.columns.values)
    for column in columns_to_add:
        points[column] = np.nan

    lon_points = np.array(points["lon"]).reshape((-1, 1))
    lat_points = np.array(points["lat"]).reshape((-1, 1))
    coords_points = np.deg2rad(np.hstack((lat_points, lon_points)))

    lon_data = np.array(szs["lon"]).reshape((-1, 1))
    lat_data = np.array(szs["lat"]).reshape((-1, 1))
    coords_data = np.deg2rad(np.hstack((lat_data, lon_data)))

    neigh = NearestNeighbors(metric="haversine", n_jobs=n_jobs)
    neigh.fit(coords_data)

    distances, indices = neigh.kneighbors(coords_points, n_neighbors=1)
    distances = distances.flatten() * EARTH_RADIUS # in km
    indices = indices.flatten()
        
    for i_points, i_szs in zip(points.index, indices):
        for column in columns_to_add:
            points.at[i_points, column] = szs.at[i_szs, column]

    points["distance_to_trench (km)"] = distances
    
    if distance_threshold is not None:
        distance_threshold = np.deg2rad(distance_threshold) * EARTH_RADIUS
        points["valid"] = distances <= distance_threshold
    # else:
    #     points["valid"] = True

    return points


def run_coregister_crustal_thickness(
    point_data: _PathOrDataFrame,
    input_dir: _PathLike,
    distance_threshold: float = 0.1,
    output_filename: Optional[_PathLike] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    
    """
    Coregister point data with gridded crustal thickness datasets across geological times.

    For each point in the dataset, this function looks up crustal thickness values
    from precomputed NetCDF grids at matching geological ages. Each point is 
    assigned a mean crustal thickness based on nearby grid cells within a given 
    search radius. If no grid cells fall within the radius, the nearest grid cell 
    is used as a fallback.

    Parameters
    ----------
    point_data : str, pd.DataFrame, or array-like
        Input dataset containing at least ["lon", "lat", "age (Ma)"].
        Can be a CSV filename, pandas DataFrame, or array-like.
    input_dir : str or Path-like
        Directory containing crustal thickness grids named
        `crustal_thickness_{time}Ma.nc` for each geological time.
    distance_threshold : float, default=0.1
        Search radius (in degrees) around each point for averaging crustal thickness
        values. Falls back to the nearest grid cell if no values are found.
    output_filename : str or Path-like, optional
        If provided, saves the results as a CSV file.
    n_jobs : int, default=1
        Number of parallel processes used to coregister multiple time steps.
        Follows joblib semantics (-1 for all CPUs).
    verbose : bool, default=False
        Print progress and diagnostic information.

    Returns
    -------
    pd.DataFrame
        The input point data with an added column:
        - crustal_thickness (m): Mean crustal thickness from nearby grid cells.

    Notes
    -----
    - Input NetCDF files must contain variables "z" (thickness), and coordinate
      variables "lon"/"lat" (or "x"/"y").
    - Distances are computed using haversine great-circle distance.
    - Each geological time in `point_data` must have a corresponding NetCDF file
      in `input_dir`.
    """

    if isinstance(point_data, str):
        point_data = pd.read_csv(point_data)
    else:
        point_data = pd.DataFrame(point_data)
    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        out = parallel(
            delayed(_coregister_crustal_thickness)(
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


def _coregister_crustal_thickness(
    time: float,
    input_dir: _PathLike,
    df: _PathOrDataFrame,
    distance_threshold: float = 0.1,
) -> pd.DataFrame:
    
    df = df.copy()
    df = df[df["age (Ma)"] == time]
    input_filename = os.path.join(
        input_dir, "crustal_thickness_{:0.0f}Ma.nc".format(time)
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
    point_lons = np.deg2rad(np.array(df["lon"]))
    point_lats = np.deg2rad(np.array(df["lat"]))
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
    df["crustal_thickness (m)"] = crustal_thickness
    
    return df


def reconstruct_points_(
    data,
    plate_reconstruction=None,
    rotation_model=None,
    static_polygons=None,
    anchor_plate_id=None,
    times=None,
    verbose=False,
):

    if plate_reconstruction is not None:
        rotation_model = plate_reconstruction.rotation_model
        static_polygons = plate_reconstruction.static_polygons
        
    if rotation_model is None:
        raise TypeError("Rotation model must be provided")

    data = _load_data(data, copy=False)
    
    if times is None:
        times = np.arange(data["age (Ma)"].round().max() + 1)
        
    times = np.sort(times)
    recon_cols = [*[f"lon_{t}" for t in times], *[f"lat_{t}" for t in times]]

    df_recon = pd.DataFrame(columns=recon_cols, index=data.index, dtype=np.float_)
    data = data.join(df_recon)

    if not isinstance(rotation_model, pygplates.RotationModel):
        rotation_model = pygplates.RotationModel(rotation_model)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        for t, subset in data.groupby("age (Ma)"):
            for i in subset.index:
                data.at[i, f"lon_{t:0.0f}"] = data.at[i, "lon"]
                data.at[i, f"lat_{t:0.0f}"] = data.at[i, "lat"]

        for t in times[::-1]:
            if verbose and t % 10 == 0:
                print(f"Reconstructing to {t:0.0f} Ma", file=stderr)
            if t == min(times):
                break
            old_lon_col = f"lon_{t:0.0f}"
            old_lat_col = f"lat_{t:0.0f}"
            new_lon_col = f"lon_{t - 1:0.0f}"
            new_lat_col = f"lat_{t - 1:0.0f}"

            subset = data[~data[old_lon_col].isna()]
            if subset.shape[0] == 0:
                continue
            lons = subset[old_lon_col]
            lats = subset[old_lat_col]

            reconstructed_points = reconstruct_points_impl(
                lons=lons,
                lats=lats,
                rotation_model=rotation_model,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
                times=t - 1,
                valid_time=(t, t),
                )
            new_lats, new_lons = reconstructed_points[0]['lats'], reconstructed_points[0]['lons']
            
            for i, new_lon, new_lat in zip(subset.index, new_lons, new_lats):
                data.at[i, new_lon_col] = new_lon
                data.at[i, new_lat_col] = new_lat
                
    return data


def extract_strain_and_rate(
        data,
        topological_model,
        anchor_plate_id,
        time_max,
        dropna=False,
        age_col="age (Ma)",
        output_filename=None,
        ):
    
    # Only copy data if necessary
    if not isinstance(data, pd.DataFrame):
        try:
            if Path(str(data)).is_file():
                data = pd.read_csv(data)
        except Exception:
            pass
        else:
            data = pd.DataFrame(data)

    if isinstance(topological_model, PlateReconstruction):
        topological_model = pygplates.TopologicalModel(
            topological_features=topological_model.topology_features,
            rotation_model=topological_model.rotation_model,
            anchor_plate_id=anchor_plate_id,
        )
    elif not isinstance(topological_model, pygplates.TopologicalModel):
        topological_model = pygplates.TopologicalModel(
            **topological_model,
            anchor_plate_id=anchor_plate_id,
            )

    if age_col not in data.columns:
        raise ValueError(f"Column '{age_col}' not found in data: {data.columns}")

    out = pd.DataFrame(
        columns=(
            "dilatation_strain_rate (/Ps)",
            "shear_strain_rate (rad/Ps)",
            "strain_style",
            "total_strain_rate (/Ps)",
            "dilatation_strain",
        ),
        index=data.index,
    )

    time_span = _get_time_span(
        topological_model=topological_model,
        time_max=time_max,
        resolution=0.5,
    )

    for time, subset in data.groupby(age_col):
        time = np.around(time)

        # Strain rate
        snapshot = topological_model.topological_snapshot(time)
        networks = snapshot.get_resolved_topologies(pygplates.ResolveTopologyType.network)
        points = [
            pygplates.PointOnSphere(lat, lon)
            for lat, lon in zip(subset["lat"], subset["lon"])
        ]
        for index, point in zip(subset.index, points):
            for network in networks:
                geom = network.get_resolved_geometry()
                if geom.is_point_in_polygon(point):
                    strain_rate = network.get_point_strain_rate(point)
                    if strain_rate is None:
                        strain_rate = pygplates.StrainRate.zero
                    out.at[index, "dilatation_strain_rate (/Ps)"] = strain_rate.get_dilatation_rate() * 1.0e15
                    out.at[index, "shear_strain_rate (rad/Ps)"] = _max_shear_rate(strain_rate.get_rate_of_deformation()) * 1.0e15
                    out.at[index, "strain_style"] = np.clip(strain_rate.get_strain_rate_style(), -1.0, 1.0)
                    out.at[index, "total_strain_rate (/Ps)"] = strain_rate.get_total_strain_rate() * 1.0e15
                    break

        # Cumulative strain
        strain_points = time_span.get_geometry_points(time)
        strains = time_span.get_strains(time)
        point_lons = []
        point_lats = []
        point_strains = []
        for p, s in zip(strain_points, strains):
            if s is None:
                continue
            d = s.get_dilatation()
            if d != 0.0:
                point_lons.append(p.to_lat_lon()[1])
                point_lats.append(p.to_lat_lon()[0])
                point_strains.append(d)
        if len(point_lons) == 0:
            continue
        x_train = pygplates.MultiPointOnSphere(
            np.column_stack((point_lats, point_lons))
        ).to_xyz_array()
        x_pred = pygplates.MultiPointOnSphere(
            points
        ).to_xyz_array()
        neigh = RadiusNeighborsRegressor(
            radius=np.sqrt(2 * (1 - np.cos(np.deg2rad(0.5))))
        )  # find all points within 0.5 arc degrees
        neigh.fit(x_train, np.array(point_strains))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cumulative_strains = neigh.predict(x_pred)
        for index, strain in zip(subset.index, cumulative_strains):
            out.at[index, "dilatation_strain"] = strain

    out = pd.concat((data, out), axis="columns")
    if dropna:
        out = out.dropna(
            subset=[
                "dilatation_strain_rate (/Ps)",
                "shear_strain_rate (rad/Ps)",
                "strain_style",
                "total_strain_rate (/Ps)",
                "dilatation_strain",
            ]
        )
        
    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        out.to_csv(output_filename, index=False)
        
    return out


def extract_strain_history(
        data,
        topological_model,
        anchor_plate_id=None,
        time_window=10,
        dropna=False,
        age_col="age (Ma)",
        columns=None,
        output_filename=None,
        ) -> pd.DataFrame:
    
    # Only copy data if necessary
    if not isinstance(data, pd.DataFrame):
        try:
            if Path(str(data)).is_file():
                data = pd.read_csv(data)
        except Exception:
            pass
        else:
            data = pd.DataFrame(data)

    if isinstance(topological_model, PlateReconstruction):
        topological_model = pygplates.TopologicalModel(
            topological_features=topological_model.topology_features,
            rotation_model=topological_model.rotation_model,
            anchor_plate_id=anchor_plate_id,
        )
    elif not isinstance(topological_model, pygplates.TopologicalModel):
        topological_model = pygplates.TopologicalModel(
            **topological_model,
            anchor_plate_id=anchor_plate_id,
            )

    if columns is None:
        columns = [
            "dilatation_strain_rate (/Ps)",
            "shear_strain_rate (rad/Ps)",
            "total_strain_rate (/Ps)",
        ]

    new_columns = []
    for column in columns:
        if column not in {
            "dilatation_strain_rate (/Ps)",
            "shear_strain_rate (rad/Ps)",
            "total_strain_rate (/Ps)",
        }:
            raise ValueError(f"Invalid column name: '{column}'")
        # Diff
        new_col = _diff_column(column)
        data[new_col] = np.nan
        new_columns.append(new_col)
        # Mean
        new_col = _mean_column(column)
        data[new_col] = np.nan
        new_columns.append(new_col)

    dts = range(time_window)
    for time in data[age_col].round().unique():
        subset = data[data[age_col].round() == time]
        values = {
            col: pd.DataFrame(
                index=subset.index,
                columns=dts,
            )
            for col in columns
        }
        initial_points = pygplates.MultiPointOnSphere(
            np.column_stack((
                subset["lat"],
                subset["lon"],
            ))
        )
        time_span = topological_model.reconstruct_geometry(
            initial_points,
            initial_time=time,
            oldest_time=time + time_window - 1,
        )

        for dt in dts:
            current_time = time + dt
            rates = time_span.get_strain_rates(current_time)
            if rates is None:
                continue
            for i, sr in zip(
                subset.index,
                time_span.get_strain_rates(current_time),
            ):
                if sr is None:
                    continue
                values["shear_strain_rate (rad/Ps)"].at[i, dt] = (
                    _max_shear_rate(sr.get_rate_of_deformation()) * 1.0e15
                )
                values["dilatation_strain_rate (/Ps)"].at[i, dt] = (
                    sr.get_dilatation_rate() * 1.0e15
                )
                values["total_strain_rate (/Ps)"].at[i, dt] = (
                    sr.get_total_strain_rate() * 1.0e15
                )

        for column in columns:
            # Diff
            diff_col = _diff_column(column)
            mean_col = _mean_column(column)
            for i, row in values[column].iterrows():
                row = row.dropna()
                if len(row) <= 1:
                    continue
                # Diff
                data.at[i, diff_col] = (
                    (row.iloc[0] - row.iloc[-1])
                    / (row.index[-1] - row.index[0])
                )
                # Mean
                data.at[i, mean_col] = row.mean()

    if dropna:
        data = data.dropna(subset=new_columns)
        
    if output_filename is not None:
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        data.to_csv(output_filename, index=False)
        
    return data


def _get_time_span(topological_model, time_max, resolution=0.5):
    
    grid_lons = np.arange(-180.0 + resolution, 180 + resolution, resolution)
    grid_lats = np.arange(-90, 90 + resolution, resolution)
    grid_lons, grid_lats = np.meshgrid(grid_lons, grid_lats)
    grid_coords = np.column_stack(
        (
            np.ravel(grid_lats),
            np.ravel(grid_lons),
        )
    )
    grid_points = pygplates.MultiPointOnSphere(grid_coords)
    grid_points = [pygplates.PointOnSphere(i) for i in grid_points]

    time_span = topological_model.reconstruct_geometry(
        grid_points,
        time_max,
        time_increment=1,
    )
    
    return time_span


def _max_shear_rate(D: np.ndarray):
    
    D = np.array(D).reshape((2, 2))
    rot_matrix = np.array((
        (0., -1.),
        (1., 0.)
    ))

    azs = np.linspace(0., np.pi, 100)
    shear_rates = np.empty(azs.shape)
    for i, az in enumerate(azs):
        n1 = np.array((np.cos(az), np.sin(az)))
        shear_rates[i] = -(n1 @ D @ (rot_matrix @ n1))
        
    return np.nanmax(shear_rates)


def _diff_column(column: str):
    
    column_split = column.split()
    if len(column_split) == 1:
        new_col = column + "_diff (/Myr)"
    else:
        new_col = (
            column_split[0] + "_diff"
            + " "
            + column_split[1].replace(")", "/Myr)")
        )
        
    return new_col


def _mean_column(column: str):
    
    column_split = column.split()
    if len(column_split) == 1:
        new_col = column + "_mean"
    else:
        new_col = (
            column_split[0] + "_mean"
            + " "
            + column_split[1]
        )
        
    return new_col


def _fill_missing_values_age(age, df, method, exclude_columns, distance_threshold, random_state):
    
    # Extract subset for the specific age
    subset = df[df["age (Ma)"] == age].copy()
    
    if distance_threshold is not None:
        count = (df["distance_to_trench (km)"] > distance_threshold).sum()
        print("Out of", len(df), "samples", count, "are located further than the", int(distance_threshold), "km distance threshold.")
        subset = subset[subset["distance_to_trench (km)"] <= distance_threshold]
    
    # Exclude specified columns from the imputation process
    subset_features = subset.drop(columns=exclude_columns)

    filled_df = pd.DataFrame()

    if method == "median":
        # Median imputation by age group
        for column in subset_features.columns:
            if subset[column].isnull().any():
                # Fill missing values with the median of the existing values
                if subset[column].dropna().empty:
                    # If the column is completely empty, fill with zeros
                    subset[column] = subset[column].fillna(0)
                else:
                    median_value = subset[column].median()
                    subset[column] = subset[column].fillna(median_value)
        filled_df = subset

    elif method == "iterative":
        # Fill columns with entirely missing values with zeros before imputation
        for column in subset_features.columns:
            if subset[column].isnull().all():
                subset[column] = 0  # Fill entire column with zeros if all values are missing
        
        # Separate features and "age (Ma)" column
        features_subset = subset.drop(columns=exclude_columns)
        
        # Initialize and apply the Iterative Imputer
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(random_state=random_state),
            random_state=random_state,
        )
        features_imputed_array = imputer.fit_transform(features_subset)

        # Update the subset with the imputed values
        imputed_df = pd.DataFrame(features_imputed_array, columns=features_subset.columns)
        # Add excluded columns back into the final result
        for col in exclude_columns:
            imputed_df[col] = subset[col].values
        filled_df = imputed_df

    else:
        raise ValueError('Invalid method. Please choose "median" or "iterative" for the imputation method.')

    return filled_df


def fill_missing_values(
        df,
        method="median",
        exclude_columns=None,
        distance_threshold=None,
        report_missing=False,
        random_state=42,
        export_to_csv=False,
        output_path="filled_data.csv",
        n_jobs=-2,
        verbose=False,
        ):

    """
    Function to fill missing values in a dataframe using either median imputation or iterative imputation, with optional parallelization.

    Parameters:
    ----------
    - df (pandas.DataFrame): Input dataframe with samples as rows and features as columns.
    - method (str): The imputation method to use. Can be "median" (default) for median imputation or "iterative" for iterative imputation using the IterativeImputer from sklearn.
    - exclude_columns (list, optional): List of columns to exclude from the imputation process (default is None).
    - distance_threshold (float, optional): Maximum distance to the trench (km) used to filter rows before imputation; if provided, only samples within this distance are considered.
    - report_missing (bool, optional): If True, prints a report showing the total number of samples and the percentage of missing values for each column (default is False).
    - random_state (int, optional): Random seed to ensure reproducibility (default is 42).
    - export_to_csv (bool, optional): If True, exports the filled dataframe to a CSV file (default is False).
    - output_path (str, optional): Path where the filled dataframe will be saved as a CSV file if export_to_csv is True (default is "filled_data.csv").
    - n_jobs (int, optional): Number of CPU cores to use for parallelization of the imputation. Defaults to -2, which means using all but two cores. Set to -1 for using all available cores.
    - verbose (bool, optional): Print progress and diagnostic information if True.

    Returns:
    ----------
    - pandas.DataFrame: A dataframe with the missing values filled based on the chosen imputation method.
    
    Notes:
    ----------
    - The function performs parallel processing to speed up the imputation process by splitting the dataframe based on the unique values of the "age (Ma)" column. Each subset is processed concurrently.
    - If the "age (Ma)" column does not exist in the dataframe, a ValueError will be raised.
    """
    
    # Default to empty list if exclude_columns is None
    if exclude_columns is None:
        exclude_columns = []
    
    # Check if "age (Ma)" column exists
    if "age (Ma)" not in df.columns:
        raise ValueError('The dataframe must have a column named "age (Ma)"')
        
    # Optional report for missing data before starting the imputation
    if report_missing:
        total_samples = len(df)
        print(f"\n{'='*40}")
        print(f"{'Missing Values Report':^40}")
        print(f"{'='*40}")
        print(f"Total number of samples: {total_samples}")
        
        # Exclude the columns specified in exclude_columns
        if 'valid' in df.columns:
            exclude_columns.append('valid')
        columns_to_report = [col for col in df.columns if col not in exclude_columns]
        
        # Calculate missing values and percentage for the columns to report
        missing_values = df[columns_to_report].isnull().sum()
        missing_percentage = (missing_values / total_samples) * 100
        
        # Filter only columns with missing values
        missing_report = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage Missing': missing_percentage
        })
        missing_report = missing_report[missing_report['Missing Values'] > 0]  # Only columns with NaNs
        
        if not missing_report.empty:
            print(f"\n{'Column':<30} {'Missing Values':<15} {'Percentage Missing':<20}")
            print("-" * 65)
            for col, row in missing_report.iterrows():
                print(f"{col:<30} {row['Missing Values']:<15} {row['Percentage Missing']:.2f}%")
        else:
            print("\nNo missing values found in the columns being reported.")
        
        print(f"{'='*40}\n")  # Add a separator for clarity
        
    # Check if there are any missing values in the dataframe
    if df.isnull().sum().sum() == 0:
        print("No missing values found. No imputation needed.")
        # Export the original dataframe to a CSV if requested
        if export_to_csv:
            df.to_csv(output_path, index=False)
        return df.copy()

    # Parallelize the processing of each age group
    unique_ages = df["age (Ma)"].unique()

    # Use joblib to parallelize the processing
    filled_dfs = Parallel(n_jobs=n_jobs, verbose=int(verbose))(delayed(_fill_missing_values_age)(age, df, method, exclude_columns, distance_threshold, random_state) for age in unique_ages)

    # Combine the results into a single dataframe
    filled_df = pd.concat(filled_dfs, ignore_index=True)

    # Export the filled dataframe to a CSV if requested
    if export_to_csv:
        filled_df.to_csv(output_path, index=False)
    
    return filled_df


def prepare_polygon_features(
    polygons_recon,
    rotation_model,
    static_polygons,
    verbose=False,
):
    
    """
    Load and partition polygons.
    """

    polygons_fc = pygplates.FeatureCollection(polygons_recon)

    needs_ids = any(
        feat.get_reconstruction_plate_id() is None
        for feat in polygons_fc
    )

    if needs_ids:

        if verbose:
            print("Partitioning polygons into plates...")

        static_fc = (
            static_polygons
            if isinstance(static_polygons, pygplates.FeatureCollection)
            else pygplates.FeatureCollection(static_polygons)
        )

        partitioned = []

        pygplates.partition_into_plates(
            static_fc,
            rotation_model,
            polygons_fc,
            partitioned,
            properties_to_copy=pygplates.PartitionIntoPlatesCopyProperties.all,
            reconstruction_time=0.0,
        )

        polygons_fc = pygplates.FeatureCollection(partitioned)

    return polygons_fc


def reconstruct_polygons(
    time,
    polygons_fc,
    rotation_model,
    output_dir,
    output_file,
    anchor_plate_id=None,
    verbose=False,
):
    
    """
    Reconstruct polygons at a given time.
    """

    os.makedirs(output_dir, exist_ok=True)

    shp_path = os.path.join(
        output_dir,
        f"{output_file}_{int(time)}Ma.shp",
    )

    if not os.path.exists(shp_path):

        if verbose:
            print(f"Reconstructing polygons at {time} Ma")

        pygplates.reconstruct(
            polygons_fc,
            rotation_model,
            shp_path,
            float(time),
            anchor_plate_id,
        )

    gdf = gpd.read_file(shp_path)
    gdf["geometry"] = gdf["geometry"].buffer(0)

    return gdf.geometry.unary_union


def filter_deposits_by_continents(
    input_csv,
    output_csv,
    continents_recon,
    plate_reconstruction,
    continents_recon_dir,
    continents_recon_file="continents",
    anchor_plate_id=None,
    verbose=False,
):

    df = pd.read_csv(input_csv)
    time_steps = df["age (Ma)"]

    rotation_model = plate_reconstruction.rotation_model
    static_polygons = plate_reconstruction.static_polygons

    # Prepare continent features
    continents_fc = prepare_polygon_features(
        continents_recon,
        rotation_model,
        static_polygons,
    )

    # Cache continent polygons
    continent_cache = {}
    keep_mask = []

    for i, row in df.iterrows():
        age = int(row["age (Ma)"])
        valid = True        
        for t in time_steps:
            if t > age:
                continue
            lon_col = f"lon_{t}"
            lat_col = f"lat_{t}"
            if lon_col not in df.columns:
                continue
            lon = row[lon_col]
            lat = row[lat_col]
            if pd.isna(lon) or pd.isna(lat):
                continue
            # Load continent polygon if needed
            if t not in continent_cache:
                continent_cache[t] = reconstruct_polygons(
                    t,
                    continents_fc,
                    rotation_model,
                    continents_recon_dir,
                    continents_recon_file,
                    anchor_plate_id,
                )

            continent_poly = continent_cache[t]
            p = Point(lon, lat)
            
            if not p.within(continent_poly):
                valid = False
                break

        keep_mask.append(valid)

        if verbose and i % 100 == 0:
            print(f"Filtering deposit {i}/{len(df)}")

    filtered_df = df[keep_mask].reset_index(drop=True)
    filtered_df.to_csv(output_csv, index=False)

    if verbose:
        print(f"Remaining deposits: {len(filtered_df)} / {len(df)}")

    return filtered_df
