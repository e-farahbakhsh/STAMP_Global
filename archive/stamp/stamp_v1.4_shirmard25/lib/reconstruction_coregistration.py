'''
Reconstruction and Coregistration

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 05/08/2025
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
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import linemerge
from sklearn.neighbors import NearestNeighbors
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
# from gplately.reconstruction import reconstruct_points
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
    output_dir: _PathLike = os.curdir,
    buffer_distance: float = 6,
    n_jobs: int = -2,
    verbose: bool = False,
    return_output: bool = False,
) -> Optional[List[gpd.GeoDataFrame]]:

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
                output_dir=output_dir,
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
    output_dir: _PathLike = os.curdir,
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
                output_dir=output_dir,
                buffer_distance=buffer_distance,
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
            )
    else:
        rotation_model = plate_reconstruction.rotation_model
        topology_features = plate_reconstruction.topology_features
        static_polygons = plate_reconstruction.static_polygons

    gplot = PlotTopologies(plate_reconstruction)
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
    projected_buffered = projected.buffer(distance_metres * direction, single_sided=True)
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
):
    
    resolved_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        [],  # Discard boundaries/networks
        float(time),
        resolved_sections,
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
        geom = linemerge(grouped.geometry.to_list())
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
    min_time=-np.inf,
    max_time=np.inf,
    n_jobs=-2,
    verbose=False,
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
        plate_reconstruction = PlateReconstruction(
            rotation_model=rotation_model,
            topology_features=topology_features,
            static_polygons=static_polygons
        )
    else:
        rotation_model = plate_reconstruction.rotation_model
        topology_features = plate_reconstruction.topology_features
        static_polygons = plate_reconstruction.static_polygons
    
    if isinstance(deposit_data, str):
        if verbose:
            print(
                "Loading deposit data from: " + deposit_data,
                file=stderr,
            )
        deposit_data = pd.read_csv(deposit_data)
    else:
        deposit_data = pd.DataFrame(deposit_data)

    deposit_data = deposit_data.drop(
        columns=["index"],
        errors="ignore",
    )

    deposit_data["age (Ma)"] = pd.to_numeric(deposit_data["age (Ma)"], errors="coerce")
    deposit_data = deposit_data[
        (deposit_data["age (Ma)"] >= min_time)
        & (deposit_data["age (Ma)"] <= max_time)
    ]

    deposit_data = _partition_and_reconstruct(
        deposit_data=deposit_data,
        plate_reconstruction=plate_reconstruction,
        time_steps=time_steps,
    )
    deposit_data = _clean_deposit_data(
        deposit_data=deposit_data,
        polygons_dir=buffer_zones_dir,
        nprocs=n_jobs,
        verbose=verbose,
    )
    deposit_data = _get_overriding_plate_ids(
        data=deposit_data,
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    
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


def _partition_and_reconstruct(deposit_data, plate_reconstruction, time_steps):
    
    time_steps = np.array(time_steps)
    
    deposit_data["age (Ma)"] = deposit_data["age (Ma)"].astype(float).apply(
        lambda x: time_steps[np.abs(time_steps - x).argmin()]
        )

    rotation_model = plate_reconstruction.rotation_model
    static_polygons = plate_reconstruction.static_polygons

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

    deposit_data["plate_id"] = np.nan
    deposit_data = deposit_data.rename(
        columns={
            "lon": "present_lon",
            "lat": "present_lat",
        }
    )
    deposit_data["lon"] = np.nan
    deposit_data["lat"] = np.nan

    partitioned = pygplates.partition_into_plates(
        partitioning_features=static_polygons,
        rotation_model=rotation_model,
        features_to_partition=features,
    )

    reconstructed = []
    times = set([i.get_valid_time()[0] for i in partitioned])
    for time in times:
        to_reconstruct = [
            i for i in partitioned if i.get_valid_time()[0] == time
        ]
        pygplates.reconstruct(
            to_reconstruct,
            rotation_model,
            reconstructed,
            time,
        )

    for i in reconstructed:
        geom = i.get_reconstructed_geometry()
        feature = i.get_feature()
        lat, lon = geom.to_lat_lon()
        index = int(feature.get_name())
        plate_id = int(feature.get_reconstruction_plate_id())

        deposit_data.at[index, "lon"] = lon
        deposit_data.at[index, "lat"] = lat
        deposit_data.at[index, "plate_id"] = plate_id
        
    return deposit_data


def _clean_deposit_data(deposit_data, polygons_dir, nprocs, verbose=False):
    
    times = deposit_data["age (Ma)"].unique()

    with Parallel(nprocs, verbose=int(verbose)) as p:
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


def _get_overriding_plate_ids(
    data,
    n_jobs=-2,
    verbose=False,
    rotation_model=None,
    topology_features=None,
    static_polygons=None,
):
    
    gdf = data.copy()
    gdf["geometry"] = [Point(lon, lat) for lon, lat in zip(gdf["lon"], gdf["lat"])]
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    times = gdf["age (Ma)"].unique()
    times_split = np.array_split(times, n_jobs)

    with Parallel(n_jobs=n_jobs, verbose=int(verbose)) as parallel:
        results = parallel(
            delayed(_overriding_plate_multiple_timesteps_parallel)(
                gdf=gdf[gdf["age (Ma)"].isin(t)],
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
            )
            for t in times_split
        )

    out = pd.concat([pd.concat(r) for r in results], ignore_index=True)
    keep_cols = set(data.columns).union({"overriding_plate_id"})
    
    return out[[col for col in out.columns if col in keep_cols]]


def _overriding_plate_multiple_timesteps_parallel(gdf, rotation_model, topology_features, static_polygons):
    
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
        topology_features=topology_features,
        rotation_model=rotation_model,
        static_polygons=static_polygons,
    )

    return _overriding_plate_multiple_timesteps(gdf, plate_reconstruction)


def _overriding_plate_multiple_timesteps(gdf, plate_reconstruction):
    
    rotation_model = plate_reconstruction.rotation_model    
    topology_features = plate_reconstruction.topology_features
    static_polygons = plate_reconstruction.static_polygons

    times = gdf["age (Ma)"].unique()
    out = []
    for time in times:
        out.append(
            _overriding_plate_timestep(
                gdf=gdf,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                time=time,
            )
        )
        
    return out


def _overriding_plate_timestep(
    gdf,
    rotation_model,
    topology_features,
    static_polygons,
    time,
):
    
    gdf = gpd.GeoDataFrame(gdf)

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
        reconstruction = PlateReconstruction(
            rotation_model=rotation_model,
            topology_features=topology_features,
            static_polygons=static_polygons,
        )
        gplot = PlotTopologies(reconstruction)
    gplot.time = time
    topologies = gplot.get_all_topologies()
    topologies.crs = "EPSG:4326"
    topologies["feature_type"] = topologies["feature_type"].astype(str)
    plate_types = {
        "gpml:TopologicalClosedPlateBoundary",
        "gpml:OceanicCrust",
        "gpml:TopologicalNetwork",
    }
    topologies = topologies[topologies["feature_type"].isin(plate_types)]

    gdf_time = gdf[gdf["age (Ma)"] == time]
    if gdf_time.crs is None:
        gdf_time.crs = topologies.crs
    joined = gdf_time.sjoin(topologies, how="inner")
    joined = joined.rename(
        columns={"reconstruction_plate_ID": "overriding_plate_id"}
    )
    
    return joined


def partition_and_reconstruct(
        deposit_data,
        plate_reconstruction,
        time_steps,
        output_filename,
        verbose=False):
    
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
            print(f"Processing deposit {index}/{len(deposit_data)}")
            
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
    threads=1,
    output_filename=None,
    seed=None,
    rotation_model=None,
    topology_features=None,
    static_polygons=None,
    verbose=False,
):

    seq = np.random.SeedSequence(entropy=seed)
    rngs = [np.random.default_rng(i) for i in seq.spawn(threads)]
    times_split = np.array_split(times, threads)

    with Parallel(threads, verbose=int(verbose)) as p:
        results = p(
            delayed(_multiple_timesteps_unlabelled)(
                times=t,
                input_dir=input_dir,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                num=num,
                rng=rng,
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
            threads=1,
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
        if time == 0.0:
            present_day_coords = pd.DataFrame(
                {
                    "lon_0": points[:, 0],
                    "lat_0": points[:, 1],
                }
            )
        else:
            present_day_coords = _reconstruct_points_present(
                data=pd.DataFrame(
                    {
                        "lon": points[:, 0],
                        "lat": points[:, 1],
                        "age (Ma)": time,
                    }
                ),
                plate_reconstruction=plate_reconstruction,
                rotation_model=rotation_model,
                static_polygons=static_polygons,
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
    n=1, output_format="radians", order="lonlat", threads=1, rng=None
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

    if threads == 1:
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
        xyz = _generate_points_threaded(n=n, threads=threads, seq=rng)

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


def _generate_points_threaded(n=1, threads=2, seq=None):
    
    seed = 42

    if seq is None:
        seq = np.random.SeedSequence(seed)
    generators = [np.random.default_rng(i) for i in seq.spawn(threads)]
    xyz = np.zeros((n, 3))

    executor = concurrent.futures.ThreadPoolExecutor(threads)
    step = np.ceil(n / threads).astype(np.int_)

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
    for i in range(threads):
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


def prepare_unlabelled_data(
    unlabelled_data,
    rotation_model,
    topology_features,
    static_polygons,
    output_filename=None,
    min_time=-np.inf,
    max_time=np.inf,
    n_jobs=1,
    verbose=False,
):
    
    if isinstance(unlabelled_data, str):
        if verbose:
            print(
                "Loading unlabelled data from file: " + unlabelled_data,
                file=stderr,
            )
        unlabelled_data = pd.read_csv(unlabelled_data)
    else:
        unlabelled_data = pd.DataFrame(unlabelled_data)

    unlabelled_data = unlabelled_data[
        (unlabelled_data["age (Ma)"] >= min_time)
        & (unlabelled_data["age (Ma)"] <= max_time)
    ]
    unlabelled_data = _get_overriding_plate_ids(
        data=unlabelled_data,
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    
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
        unlabelled_data.to_csv(output_filename, index=False)
    
    return unlabelled_data


def generate_grid_points(
    times,
    resolution,
    polygons_dir,
    rotation_model=None,
    topology_features=None,
    static_polygons=None,
    output_filename=None,
    n_jobs=1,
    verbose=False,
):
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
    verbose=False,
):
    
    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
    )

    out = [
        _grid_points_time(
            time=t,
            resolution=resolution,
            polygons_dir=polygons_dir,
            plate_reconstruction=plate_reconstruction,
        )
        for t in times
    ]
    return pd.concat(out, ignore_index=True)


def _grid_points_time(
    time,
    resolution,
    polygons_dir,
    plate_reconstruction,
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
                times=t - 1,
                valid_time=(t, t),
                )
            new_lats, new_lons = reconstructed_points[0]['lats'], reconstructed_points[0]['lons']
            
            for i, new_lon, new_lat in zip(subset.index, new_lons, new_lats):
                data.at[i, new_lon_col] = new_lon
                data.at[i, new_lat_col] = new_lat
                
    return data
