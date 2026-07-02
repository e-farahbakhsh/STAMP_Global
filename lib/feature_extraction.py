'''
Feature Extraction

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 04/05/2026
'''


import os
import re
from sys import stderr
from typing import (
    Iterable,
    Literal,
    Optional,
    Sequence,
    Union,
)
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pygplates
from rasterio.enums import MergeAlg
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
import xarray as xr

import gplately
from gplately import (
    EARTH_RADIUS,
    PlateReconstruction,
    PlotTopologies,
    Raster,
)
from gplately.tools import plate_isotherm_depth

from melt import katz_2003 as mlt


_PathLike = Union[os.PathLike, str]
_PathOrDataFrame = Union[_PathLike, pd.DataFrame]
_FeatureCollectionInput = Union[
    pygplates.Feature,
    pygplates.FeatureCollection,
    str,
    Iterable[pygplates.Feature],
    Iterable[
        Union[
            pygplates.Feature,
            pygplates.FeatureCollection,
            str,
            Iterable[pygplates.Feature],
        ]
    ],
]
_RotationModelInput = Union[
    pygplates.RotationModel,
    _FeatureCollectionInput,
]


def run_calculate_convergence(
    min_time: float,
    max_time: float,
    temporal_resolution: int,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[Union[Sequence[str], str]] = None,
    topology_features: Optional[Sequence[str]] = None,
    static_polygons: Optional[Sequence[str]] = None,
    anchor_plate_id: int = None,
    output_filename: Optional[str] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    
    """
    Calculate convergence parameters along global subduction zones 
    over a specified geological time range.
    
    This function reconstructs plate boundary topologies at discrete
    time steps, tessellates subduction zones into regularly spaced
    points, and computes kinematic quantities such as convergence
    rate, obliquity, trench velocity, and subducting plate velocities.
    Distances along trenches are also reported.
    
    Results are returned as a concatenated pandas DataFrame and can 
    optionally be saved to CSV.
    
    Parameters
    ----------
    min_time : float
        Start time of the reconstruction interval (Ma).
    max_time : float
        End time of the reconstruction interval (Ma).
    temporal_resolution : int
        Time step increment between reconstructions (Myr).
    plate_reconstruction : PlateReconstruction, optional
        Pre-constructed PlateReconstruction object. If not provided,
        one will be created internally from the input files.
    rotation_model : str or sequence of str, optional
        Rotation model file(s) describing plate motions.
    topology_features : sequence of str, optional
        Topological feature collection(s) (e.g., plate boundaries).
    static_polygons : sequence of str, optional
        Static polygon file(s) used for plate partitioning.
    anchor_plate_id : int
        Anchor plate ID for reconstruction.
    output_filename : str, optional
        If provided, save the DataFrame to this CSV file.
    n_jobs : int, default=1
        Number of parallel jobs. Parallel execution requires that
        topology and rotation filenames are provided.
    verbose : bool, default=False
        Print progress messages if True.
    
    Returns
    -------
    pd.DataFrame
        Table of tessellated subduction zone points containing:
        - Longitude, latitude
        - Convergence rate and obliquity
        - Trench velocity and orientation
        - Subducting and overriding plate IDs
        - Distances along trenches (km)
        - Orthogonal/parallel velocity components
        - Geological age (Ma)
    
    Notes
    -----
    Internally uses `PlateReconstruction.tessellate_subduction_zones` 
    to sample subduction trenches and compute kinematic statistics .
    Distances originally computed in degrees are converted to km 
    using Earth's mean radius.
    """
    
    use_parallel_func_from_files = False

    if plate_reconstruction is None:
        if n_jobs > 1:
            use_parallel_func_from_files = True
        else:
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
            )

    times = np.arange(min_time, max_time + temporal_resolution, temporal_resolution)

    if n_jobs == 1:
        data = [
            _tessellate_szs(
                plate_reconstruction=plate_reconstruction,
                time=t,
                ignore_warnings=True,
            )
            for t in times
        ]
    else:
        if not use_parallel_func_from_files:
            raise RuntimeError(
                "Parallel execution requires `topology_filenames` and `rotation_filenames`."
            )
        with Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0) as parallel:
            data = parallel(
                delayed(_tessellate_szs_parallel)(
                    rotation_model=rotation_model,
                    topology_features=topology_features,
                    static_polygons=static_polygons,
                    anchor_plate_id=anchor_plate_id,
                    time=t,
                    ignore_warnings=True,
                )
                for t in times
            )

    data = pd.concat(data)

    for col in (
        "distance_to_trench_edge (degrees)",
        "distance_from_trench_start (degrees)",
    ):
        if col in data.columns:
            x_km = np.deg2rad(data[col]) * EARTH_RADIUS
            data[col.replace("(degrees)", "(km)")] = x_km
            data = data.drop(columns=col, errors="ignore")
            
    # Save to CSV if a path is provided
    if output_filename:
        data.to_csv(output_filename, index=False)
        if verbose:
            print(f"Results written to: {output_filename}")

    return data


def _tessellate_szs(
    plate_reconstruction: PlateReconstruction,
    time: float,
    tessellation_threshold_radians: float = 0.001,
    ignore_warnings: bool = True,
) -> pd.DataFrame:
    
    data = plate_reconstruction.tessellate_subduction_zones(
        time=time,
        tessellation_threshold_radians=tessellation_threshold_radians,
        ignore_warnings=ignore_warnings,
        output_distance_to_nearest_edge_of_trench=True,
        output_distance_to_start_edge_of_trench=True,
        output_convergence_velocity_components=True,
        output_trench_absolute_velocity_components=True,
        output_subducting_absolute_velocity=True,
        output_subducting_absolute_velocity_components=True,
    )
    column_names = (
        "lon",
        "lat",
        "convergence_rate (cm/yr)",
        "convergence_obliquity (degrees)",
        "trench_velocity (cm/yr)",
        "trench_velocity_obliquity (degrees)",
        "arc_segment_length (degrees)",
        "trench_normal_angle (degrees)",
        "subducting_plate_ID",
        "trench_plate_ID",
        "distance_to_trench_edge (degrees)",
        "distance_from_trench_start (degrees)",
        "convergence_rate_orthogonal (cm/yr)",
        "convergence_rate_parallel (cm/yr)",
        "trench_velocity_orthogonal (cm/yr)",
        "trench_velocity_parallel (cm/yr)",
        "subducting_plate_absolute_velocity (cm/yr)",
        "subducting_plate_absolute_obliquity (degrees)",
        "subducting_plate_absolute_velocity_orthogonal (cm/yr)",
        "subducting_plate_absolute_velocity_parallel (cm/yr)",
    )
    out = pd.DataFrame(
        data,
        columns=column_names,
    )
    out["age (Ma)"] = np.float64(time)
        
    return out


def _tessellate_szs_parallel(
    rotation_model: Union[Sequence[str], str],
    topology_features: Sequence[str],
    static_polygons: Sequence[str],
    time: float,
    anchor_plate_id: int = None,
    tessellation_threshold_radians: float = 0.001,
    ignore_warnings: bool = True,
) -> pd.DataFrame:
    
    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
        anchor_plate_id=anchor_plate_id,
    )
    
    return _tessellate_szs(
        plate_reconstruction=plate_reconstruction,
        time=time,
        tessellation_threshold_radians=tessellation_threshold_radians,
        ignore_warnings=ignore_warnings,
    )


def run_coregister_ocean_rasters(
    times: Sequence[float],
    input_data: Union[_PathLike, Sequence[pd.DataFrame]],
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    static_polygons: Optional[_FeatureCollectionInput] = None,
    anchor_plate_id: int = None,
    plates_dir: Optional[_PathLike] = None,
    agegrid_dir: Optional[_PathLike] = None,
    spreadrate_dir: Optional[_PathLike] = None,
    sedthick_dir: Optional[_PathLike] = None,
    carbonate_dir: Optional[_PathLike] = None,
    co2_dir: Optional[_PathLike] = None,
    output_filename: Optional[str] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    
    """
    Co-register seafloor and other oceanic raster data with subduction
    zone geometries across multiple geological times.
    
    This function associates trench tessellation
    points (from a convergence calculation) with values from
    time-dependent global rasters such as seafloor age, spreading
    rate, sediment thickness, carbonate thickness, and crustal CO₂
    storage. Each point is tagged with the properties of its host
    oceanic plate at the specified reconstruction time.
    
    The workflow:
      1. Load trench point data for the requested times.
      2. Generate or read global plate ID maps for each time step.
      3. Load raster datasets (agegrid, spreading rate, sediment
         thickness, carbonate, CO₂) corresponding to each time.
      4. For each subducting plate ID, spatially match trench points
         to nearby raster cells belonging to that plate.
      5. Attach interpolated mean raster values to the trench point
         DataFrame.
      6. Concatenate all times into a single output table.
    
    Parameters
    ----------
    times : sequence of float
        Geological times (Ma) to process.
    input_data : str, or pandas.DataFrame, or sequence of DataFrames
        Input trench tessellation data. Can be a CSV filename or
        already-loaded DataFrame(s) from `run_calculate_convergence`.
    plate_reconstruction : PlateReconstruction, optional
        Pre-constructed reconstruction object. If None, one is
        created internally from the input rotation/topology files.
    rotation_model : str, sequence of str, or pygplates.RotationModel, optional
        Rotation model defining plate motions.
    topology_features : str, sequence, or pygplates.FeatureCollection, optional
        Plate boundary feature collection(s).
    static_polygons : str, sequence, or pygplates.FeatureCollection, optional
        Static polygon feature collection(s) for partitioning plates.
    anchor_plate_id : int
        Anchor plate ID for reconstruction.
    plates_dir : str, optional
        Directory containing precomputed plate ID rasters
        (`plate_ids_{time}Ma.nc`). If not provided, they are generated
        on the fly.
    agegrid_dir : str, optional
        Directory containing seafloor age rasters
        (`seafloor_age_{time}Ma.nc`).
    spreadrate_dir : str, optional
        Directory with spreading rate rasters. If None, defaults to
        `agegrid_dir`. If both are None, spreading rates are skipped.
    sedthick_dir : str, optional
        Directory with sediment thickness rasters.
    carbonate_dir : str, optional
        Directory with carbonate thickness rasters. For ages >170 Ma,
        a zero-thickness placeholder file is used.
    co2_dir : str, optional
        Directory with crustal CO₂ density rasters.
    output_filename : str, optional
        If provided, results are written to this CSV file.
    n_jobs : int, default=1
        Number of parallel processes to use. Parallel execution
        splits times and input data across workers.
    verbose : bool, default=False
        Print progress messages if True.
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing, for each trench point and time:
        - Geographic coordinates
        - Subducting plate ID
        - Seafloor age (Ma)
        - Spreading rate (km/Myr)
        - Sediment thickness (m)
        - Carbonate thickness (m)
        - Crustal carbon density (t/m²)
        - Original convergence parameters from input_data
    
    Notes
    -----
    * Raster values are assigned by finding nearby grid cells within
      the same subducting plate ID; if no nearby cells are found, the
      nearest available cell is used.
    * Plate ID rasters are generated using `PlotTopologies` if not
      supplied via `plates_dir`.
    * This function is designed for workflows combining tectonic
      reconstructions with oceanic property datasets to analyse
      subduction-related fluxes through time.
    """

    if isinstance(input_data, str):
        input_data = pd.read_csv(input_data)
    if isinstance(input_data, pd.DataFrame):
        input_data = [
            (input_data[input_data["age (Ma)"] == time]).copy()
            for time in times
        ]

    if n_jobs == 1:
        out = _coregister_ocean_rasters_subset(
            times=times,
            dfs=input_data,
            plate_reconstruction=plate_reconstruction,
            topology_features=topology_features,
            rotation_model=rotation_model,
            static_polygons=static_polygons,
            anchor_plate_id=anchor_plate_id,
            plates_dir=plates_dir,
            agegrid_dir=agegrid_dir,
            spreadrate_dir=spreadrate_dir,
            sedthick_dir=sedthick_dir,
            carbonate_dir=carbonate_dir,
            co2_dir=co2_dir,
        )
    else:
        times_split = np.array_split(times, n_jobs)
        df_array = np.empty(len(input_data), dtype="object")
        for i, df in enumerate(input_data):
            df_array[i] = df
        input_data_split = np.array_split(df_array, n_jobs)

        with Parallel(n_jobs, verbose=int(verbose)) as parallel:
            results = parallel(
                delayed(_coregister_ocean_rasters_subset_parallel)(
                    times=t,
                    dfs=d,
                    rotation_model=rotation_model,
                    topology_features=topology_features,
                    static_polygons=static_polygons,
                    anchor_plate_id=anchor_plate_id,
                    plates_dir=plates_dir,
                    agegrid_dir=agegrid_dir,
                    spreadrate_dir=spreadrate_dir,
                    sedthick_dir=sedthick_dir,
                    carbonate_dir=carbonate_dir,
                    co2_dir=co2_dir,
                )
                for t, d in zip(times_split, input_data_split)
            )
        out = []
        for i in results:
            out.extend(i)

    out = pd.concat(out, ignore_index=True)
    
    # Save to CSV if a path is provided
    if output_filename:
        out.to_csv(output_filename, index=False)
        if verbose:
            print(f"Results written to: {output_filename}")
        
    return out


def _coregister_ocean_rasters_subset(
    times,
    dfs,
    agegrid_dir,
    spreadrate_dir,
    sedthick_dir,
    carbonate_dir,
    co2_dir,
    output_dir,
    plate_reconstruction=None,
    rotation_model=None,
    topology_features=None,
    static_polygons=None,
    anchor_plate_id: int = None,
    plates_dir=None,
    **kwargs,
):
    
    if plates_dir is None and plate_reconstruction is None:
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

    return [
        _coregister_ocean_rasters(
            time=t,
            df=df,
            rotation_model=rotation_model,
            topology_features=topology_features,
            static_polygons=static_polygons,
            anchor_plate_id=anchor_plate_id,
            plates_dir=plates_dir,
            agegrid_dir=agegrid_dir,
            spreadrate_dir=spreadrate_dir,
            sedthick_dir=sedthick_dir,
            carbonate_dir=carbonate_dir,
            co2_dir=co2_dir,
            **kwargs,
        )
        for t, df in zip(times, dfs)
    ]


def _coregister_ocean_rasters_subset_parallel(
    times,
    dfs,
    agegrid_dir,
    spreadrate_dir,
    sedthick_dir,
    carbonate_dir,
    co2_dir,
    rotation_model=None,
    topology_features=None,
    static_polygons=None,
    anchor_plate_id: int = None,
    plates_dir=None,
    **kwargs,
):
    
    if plates_dir is None:
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

    return [
        _coregister_ocean_rasters(
            time=t,
            df=df,
            plate_reconstruction=plate_reconstruction,
            rotation_model=rotation_model,
            topology_features=topology_features,
            static_polygons=static_polygons,
            anchor_plate_id=anchor_plate_id,
            plates_dir=plates_dir,
            agegrid_dir=agegrid_dir,
            spreadrate_dir=spreadrate_dir,
            sedthick_dir=sedthick_dir,
            carbonate_dir=carbonate_dir,
            co2_dir=co2_dir,
            **kwargs,
        )
        for t, df in zip(times, dfs)
    ]


def _coregister_ocean_rasters(
    time: float,
    df: _PathOrDataFrame,
    agegrid_dir: _PathLike,
    spreadrate_dir: _PathLike,
    sedthick_dir: _PathLike,
    carbonate_dir: _PathLike,
    co2_dir: _PathLike,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    static_polygons: Optional[_FeatureCollectionInput] = None,
    anchor_plate_id: int = None,
    plates_dir: Optional[_PathLike] = None,
    subducted_thickness_dir: Optional[_PathLike] = None,
    subducted_sediments_dir: Optional[_PathLike] = None,
    subducted_carbonates_dir: Optional[_PathLike] = None,
    subducted_water_dir: Optional[_PathLike] = None,
    **kwargs,
) -> pd.DataFrame:
    
    if isinstance(df, str):
        df = pd.read_csv(df)
    else:
        df = pd.DataFrame(df)

    if plates_dir is None:
        raster = _create_plate_map(
            time=time,
            plate_reconstruction=plate_reconstruction,
            rotation_model=rotation_model,
            topology_features=topology_features,
            static_polygons=static_polygons,
            anchor_plate_id=anchor_plate_id,
            **kwargs,
        )
        plates = np.array(raster)
    else:
        plates_filename = os.path.join(
            plates_dir,
            "plate_ids_{}Ma.nc".format(time),
        )
        plates = np.array(Raster(plates_filename))
    plates[np.isnan(plates)] = -1
    plates = plates.astype(np.int_)

    if agegrid_dir is None:
        agegrid_filename = None
    else:
        agegrid_filename = os.path.join(
            agegrid_dir, f"seafloor_age_{time:0.0f}Ma.nc"
        )
        if not os.path.isfile(agegrid_filename):
            raise FileNotFoundError(
                "Age grid file not found: " + agegrid_filename
            )

    if spreadrate_dir is None:
        spreadrate_filename = None
    elif spreadrate_dir == agegrid_dir:
        spreadrate_filename = os.path.join(
            spreadrate_dir, f"seafloor_age_{time:0.0f}Ma.nc"
        )
    else:
        spreadrate_filename = os.path.join(
            spreadrate_dir, f"spreading_rate_{time:0.0f}Ma.nc"
        )
    if spreadrate_filename is not None and not os.path.isfile(spreadrate_filename):
        raise FileNotFoundError(
            "Spreading rate file not found: " + spreadrate_filename
        )

    if sedthick_dir is None:
        sedthick_filename = None
    else:
        sedthick_filename = os.path.join(
            sedthick_dir, f"sediment_thickness_{time:0.0f}Ma.nc"
        )
        if not os.path.isfile(sedthick_filename):
            raise FileNotFoundError(
                "Sediment thickness file not found: " + sedthick_filename
            )

    if co2_dir is None:
        co2_filename = None
    else:
        co2_filename = os.path.join(
            co2_dir,
            "crustal_co2_{}Ma.nc".format(time),
        )
        if not os.path.isfile(co2_filename):
            raise FileNotFoundError(
                "Crustal CO2 file not found: " + co2_filename
            )
            
    if carbonate_dir is None:
        carbonate_filename = None
    elif time > 170:
        carbonate_filename = os.path.join(
            carbonate_dir, "carbonate_thickness_zero.nc"
        )
    else:
        carbonate_filename = os.path.join(
            carbonate_dir, "carbonate_thickness_{}Ma.nc".format(time)
        )
        if not os.path.isfile(carbonate_filename):
            raise FileNotFoundError(
                "Carbonate thickness file not found: " + carbonate_filename
            )

    df["seafloor_age (Myr)"] = np.nan
    df["age (Ma)"] = time

    raster_data = {}
    for filename, name in zip(
        (
            agegrid_filename,
            spreadrate_filename,
            sedthick_filename,
            carbonate_filename,
            co2_filename,
        ),
        (
            "agegrid",
            "spreadrate",
            "sedthick",
            "carbonate",
            "co2",
        ),
    ):
        if filename is None:
            continue
        raster_data[name] = {}
        with xr.open_dataset(filename) as dset:
            if name == "agegrid" and "seafloor_age" in dset.data_vars:
                varname = "seafloor_age"
            elif name == "spreadrate" and "spreading_rate" in dset.data_vars:
                varname = "spreading_rate"
            else:
                varname = "z"
            raster = np.array(dset[varname])
            try:
                lon = np.array(dset["lon"])
            except KeyError:
                lon = np.array(dset["x"])
            try:
                lat = np.array(dset["lat"])
            except KeyError:
                lat = np.array(dset["y"])

        if raster.shape != plates.shape:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raster = resize(raster, plates.shape, order=1, mode="wrap")
            lon = np.linspace(lon.min(), lon.max(), raster.shape[1])
            lat = np.linspace(lat.min(), lat.max(), raster.shape[0])

        raster_data[name]["data"] = raster
        raster_data[name]["lon"] = lon
        raster_data[name]["lat"] = lat

    column_names = {
        "agegrid": "seafloor_age (Myr)",
        "spreadrate": "seafloor_spreading_rate (km/Myr)",
        "sedthick": "sediment_thickness (m)",
        "carbonate": "carbonate_thickness (m)",
        "co2": "crustal_carbon_density (t/m^2)",
    }
    for plate_id in df["subducting_plate_ID"].unique():
        df_plate = df[df["subducting_plate_ID"] == plate_id]
        lon_points = np.array(df_plate["lon"]).reshape((-1, 1))
        lat_points = np.array(df_plate["lat"]).reshape((-1, 1))
        coords_points = np.deg2rad(np.hstack((lat_points, lon_points)))

        for name in raster_data:
            raster = raster_data[name]["data"]
            column_name = column_names[name]
            plate_mask = np.logical_and(plates == plate_id, ~np.isnan(raster))
            if plate_mask.sum() == 0:
                continue
            raster_plate = raster[plate_mask].flatten()
            lon_data, lat_data = np.meshgrid(
                raster_data[name]["lon"],
                raster_data[name]["lat"],
            )
            lon_data = lon_data[plate_mask].flatten().reshape((-1, 1))
            lat_data = lat_data[plate_mask].flatten().reshape((-1, 1))
            coords_data = np.deg2rad(np.hstack((lat_data, lon_data)))
            neigh = NearestNeighbors(metric="haversine", n_jobs=1, radius=0.001)
            neigh.fit(coords_data)            
            mean_values = []
            
            for point in coords_points:
                distances, indices = neigh.radius_neighbors(
                    point.reshape(1, -1), return_distance=True
                )
                point_indices = indices[0]
                if len(point_indices) == 0:
                    dist, idx = neigh.kneighbors(
                        point.reshape(1, -1), n_neighbors=1, return_distance=True
                    )
                    mean_value = raster_plate[idx[0][0]]
                else:
                    mean_value = np.mean(raster_plate[point_indices])
                mean_values.append(mean_value)
            df.loc[df["subducting_plate_ID"] == plate_id, column_name] = mean_values

    return df


def _create_plate_map(
    time: float,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[pygplates.RotationModel] = None,
    topology_features: Optional[pygplates.FeatureCollection] = None,
    static_polygons: Optional[pygplates.FeatureCollection] = None,
    anchor_plate_id: int = None,
    resolution: float = 0.1,  # degrees
    tessellate_degrees: Optional[float] = None,
    output_filename: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = False,
) -> Raster:
    
    """
    Create a global plate id raster for a reconstruction time.

    Strategy:
      1) Rasterize reconstructed topologies (fast).
      2) If any fill (-1) remains, fall back to pyGPlates PlatePartitioner
         evaluated at the SAME pixel centers as the rasterize grid (slow but robust).

    Output:
      Always returns Raster with extent="global" and origin="lower" on the same
      (ny, nx) grid implied by from_bounds(minx, miny, maxx, maxy, nx, ny).
    """

    time = float(time)
    resolution = float(resolution)
    if tessellate_degrees is None:
        tessellate_degrees = resolution
    tessellate_degrees = float(tessellate_degrees)

    # Ensure a PlateReconstruction object
    if not isinstance(plate_reconstruction, PlateReconstruction):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
            )

    # Ensure a pygplates.RotationModel for the partitioner fallback
    if rotation_model is not None and not isinstance(rotation_model, pygplates.RotationModel):
        rotation_model = pygplates.RotationModel(rotation_model)
    else:
        # PlateReconstruction stores one, but keep behavior conservative.
        # If user passed None here, PlatePartitioner fallback will error anyway,
        # so we only normalize when provided.
        rotation_model = rotation_model

    # Build topologies
    gplot = PlotTopologies(plate_reconstruction, anchor_plate_id=anchor_plate_id)
    gplot.time = time
    topologies = gplot.get_all_topologies(tessellate_degrees=tessellate_degrees)
    topologies["feature_type"] = topologies["feature_type"].astype(str)

    # Drop null geometries early
    topologies = topologies[topologies["geometry"].notnull()].copy()

    # Sorting
    type_rank_map = {
        "gpml:TopologicalNetwork": 0,
        "gpml:OceanicCrust": 1,
        "gpml:TopologicalClosedPlateBoundary": 2,
    }
    topologies["type_rank"] = topologies["feature_type"].map(type_rank_map).fillna(99).astype(int)
    # Area is only used for ordering
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Geometry is in a geographic CRS.*",
            category=UserWarning,
        )
        topologies["geom_area"] = topologies["geometry"].area

    topologies = topologies.sort_values(
        ["type_rank", "geom_area"],
        ascending=[True, False],  # large first within type
    )

    # Materialize shapes
    shapes = list(zip(
        topologies["geometry"].to_list(),
        topologies["reconstruction_plate_ID"].to_list(),
    ))

    # Grid definition
    minx, maxx, miny, maxy = -180.0, 180.0, -90.0, 90.0
    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)
    nx = lons.size
    ny = lats.size
    transform = from_bounds(minx, miny, maxx, maxy, nx, ny)

    # Fast path: rasterize topologies
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        grid = rasterize(
            shapes=shapes,
            out_shape=(ny, nx),
            fill=-1,
            dtype=np.int32,
            merge_alg=MergeAlg.replace,
            transform=transform,
        )

    # rasterio output is upper-left origin; convert to lower-left for Raster
    grid = np.flipud(grid)

    raster = Raster(grid, extent="global", origin="lower")

    # Cheap fill check
    if (grid == -1).any():
        if rotation_model is None:
            raise ValueError(
                "PlatePartitioner fallback required (fill=-1 present), but rotation_model is None."
            )

        # Assemble partitioning features
        partition_features = []
        if topology_features is not None:
            partition_features.extend(
                pygplates.FeaturesFunctionArgument(topology_features).get_features()
            )
        if static_polygons is not None:
            partition_features.extend(
                pygplates.FeaturesFunctionArgument(static_polygons).get_features()
            )
        if not partition_features:
            raise ValueError(
                "PlatePartitioner fallback required (fill=-1 present), but no topology_features/static_polygons provided."
            )

        # Version-safe PlatePartitioner construction
        try:
            partitioner = pygplates.PlatePartitioner(
                partition_features, rotation_model, reconstruction_time=time
            )
        except TypeError:
            partitioner = pygplates.PlatePartitioner(
                partition_features, rotation_model, time
            )

        # Evaluate partitioner at the SAME pixel centers implied by from_bounds()
        dx = (maxx - minx) / nx
        dy = (maxy - miny) / ny
        lon_centers = minx + (np.arange(nx) + 0.5) * dx
        lat_centers = miny + (np.arange(ny) + 0.5) * dy
        lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)

        plate_ids = np.full((ny, nx), -1, dtype=np.int32)

        # Iterate through pixel centers (slow but consistent with the rasterize grid)
        for j in range(ny):
            for i in range(nx):
                pt = pygplates.PointOnSphere(lat2d[j, i], lon2d[j, i])
                plate = partitioner.partition_point(pt)
                if plate:
                    plate_ids[j, i] = plate.get_feature().get_reconstruction_plate_id()

        # plate_ids already uses increasing lat_centers (south->north), i.e. lower-left origin,
        # matching Raster(origin="lower") without flipping.
        raster = Raster(plate_ids, extent="global", origin="lower")

    if output_filename is not None:
        if verbose:
            print(
                " - Writing output file: " + os.path.basename(output_filename),
                file=stderr,
                flush=True,
            )
        raster.save_to_netcdf4(output_filename)

    return raster


def calculate_carbon(df, inplace=False):
    
    if (not inplace) or (not isinstance(df, pd.DataFrame)):
        if isinstance(df, str):
            df = pd.read_csv(df)
        else:
            df = pd.DataFrame(df)
    seds_thickness = df["carbonate_thickness (m)"]
    seds = (
        seds_thickness
        * 0.7  # Average CO3 in carbonate rock
        * 0.41  # Pore space
        * 2710.0  # Density
        * 12.0/100.1  # CaCO3 to C
        * 1.0e-3  # kg/m2 to t/m2
    )
    crust = df["crustal_carbon_density (t/m^2)"]
    df["carbonate_carbon_density (t/m^2)"] = seds
    df["total_carbon_density (t/m^2)"] = seds + crust
    
    return df


def calculate_slab_flux(df, inplace=False):

    if isinstance(df, str):
        df = pd.read_csv(df)
    if not inplace:
        df = df.copy()

    rates = np.array(df["convergence_rate_orthogonal (cm/yr)"]) * 0.01
    if "plate_thickness (m)" in df.columns.values:
        thicknesses = np.array(df["plate_thickness (m)"])
    else:
        ages = np.array(df["seafloor_age (Myr)"])
        thicknesses = plate_isotherm_depth(ages, maxiter=100)
    slab_flux = rates * thicknesses
    df["slab_flux (m^2/yr)"] = slab_flux
    
    return df


def extract_subducted_thickness(
    data: Union[pd.DataFrame, Sequence[pd.DataFrame]],
    columns: Union[None, Literal["default"], str, Sequence[str]] = None,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[Union[pygplates.RotationModel, Sequence[_PathLike], _PathLike]] = None,
    topology_features: Optional[Union[pygplates.FeatureCollection, Sequence[_PathLike], _PathLike]] = None,
    static_polygons: Optional[Union[pygplates.FeatureCollection, Sequence[_PathLike], _PathLike]] = None,
    anchor_plate_id: int = None,
    grid_resolution: float = 0.5,
    method: Literal["nearest", "linear"] = "nearest",
    time_lag: float = 0.0,
    n_jobs: int = -2,
    verbose: bool = False,
) -> pd.DataFrame:
    
    """
    Estimate cumulative volumes of subducted oceanic material
    (sediments, crust, carbonates, water, carbon) on a global grid
    through geological time.

    This function takes trench tessellation data (e.g., from
    `run_coregister_ocean_rasters`) and converts thickness values
    into subducted volumes by combining:
      * material thickness at the trench,
      * trench segment length, and
      * orthogonal convergence rate.

    At each time step, subducted volume fluxes (m³/Myr) are calculated,
    accumulated over time, and gridded at the specified resolution.
    The resulting raster volumes are then interpolated back to the
    trench points as new columns.

    Parameters
    ----------
    data : pandas.DataFrame
        Input table with trench tessellation results containing:
        - `age (Ma)`
        - `lon`, `lat`
        - `arc_segment_length (degrees)`
        - `convergence_rate_orthogonal (cm/yr)`
        - Thickness columns such as:
          `sediment_thickness (m)`,
          `plate_thickness (m)`,
          `carbonate_thickness (m)`,
          `total_water_thickness (m)`,
          `total_carbon_density (t/m^2)`
    columns : str, list of str, or "default", optional
        Which columns to process. If `"default"` or None, only columns
        present in `data` are used from the standard set above.
    plate_reconstruction : PlateReconstruction, optional
        If provided, generates plate ID rasters to ensure subducted
        volumes are correctly partitioned by plate.
    rotation_model : RotationModel, str, or sequence of str, optional
        Rotation model or rotation model file(s) describing plate motions.
    topology_features : FeatureCollection, str, or sequence of str, optional
        Topological feature collection(s) (e.g., plate boundaries) or file(s).
    static_polygons : FeatureCollection, str, or sequence of str, optional
        Static polygon(s) or file(s) used for plate partitioning.
    anchor_plate_id : int
        Anchor plate ID for reconstruction.
    grid_resolution : float, default=0.5
        Spatial resolution of the output grid in degrees.
    method : str, default="nearest"
        Interpolation method for mapping gridded volumes back to trench
        points. Options include `nearest` or `linear` depending on
        the Raster class.
    time_lag : float, default=0.0
        Length of the backward-looking time window in Myr used to
        accumulate subducted material for each trench point.

        Example:
        - trench point age = 10 Ma
        - time_lag = 10 Myr

        Then the returned value is the sum of the gridded subduction
        rates from 20 Ma to 10 Ma, not from the start of the
        reconstruction to 10 Ma.
    n_jobs : int, default=-2
        Number of parallel processes to use. Parallel execution
        splits times and input data across workers.
    verbose : bool, default=False
        Print progress messages if True.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with additional columns giving lag-windowed
        subducted volumes/densities at each trench point, such as:
        - `subducted_sediment_volume (m)`
        - `subducted_plate_volume (m)`
        - `subducted_carbonate_volume (m)`
        - `subducted_water_volume (m)`
        - `subducted_carbon_density (t/m^2)`

    Notes
    -----
    * Subduction flux is calculated as:

        volume_rate = thickness × trench_length × convergence_rate

      where:
        - thickness is in meters,
        - trench length is converted from arc length (degrees) to meters,
        - convergence rate is converted from cm/yr to m/Myr.

    * The gridded flux is normalised by cell area, so the stored grids
      are equivalent thickness-rate grids (m/Myr for thickness inputs,
      t/m²/Myr for carbon-density inputs).
    * If `time_lag > 0`, each trench point samples only the subduction
      accumulated within the previous `time_lag` Myr window.
    """

    if columns == "default" or columns is None:
        columns = [
            i for i in
            [
                "sediment_thickness (m)",
                "plate_thickness (m)",
                "carbonate_thickness (m)",
                "total_water_thickness (m)",
                "crustal_carbon_density (t/m^2)",
                "carbonate_carbon_density (t/m^2)",
                "total_carbon_density (t/m^2)",
            ]
            if i in data.columns
        ]
    elif isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)

    times = np.sort(data["age (Ma)"].unique())[::-1]
    grids = {i: [] for i in columns}

    xedges = np.arange(-180.0, 180.0 + grid_resolution, grid_resolution)
    glons = (0.5 * (np.roll(xedges, 1) + xedges))[1:]
    yedges = np.arange(-90.0, 90.0 + grid_resolution, grid_resolution)
    glats = (0.5 * (np.roll(yedges, 1) + yedges))[1:]

    mlons, mlats = np.meshgrid(glons, glats)
    lon_lengths = _longitude_length(mlats, delta=grid_resolution)
    lat_lengths = np.full_like(mlats, _latitude_length(delta=grid_resolution))
    cell_areas = lon_lengths * lat_lengths

    for time in times:
        subset = data[data["age (Ma)"] == time]
        for column in columns:
            # Thickness in m, or carbon density in t/m^2
            thickness = np.array(subset[column])

            # Trench segment length in m
            segment_length = (
                np.deg2rad(np.array(subset["arc_segment_length (degrees)"]))
                * EARTH_RADIUS
                * 1000.0
            )

            # Rate of subduction in m/Myr
            subduction_rate = (
                np.array(subset["convergence_rate_orthogonal (cm/yr)"])
                * 0.01
                * 1.0e6
            )

            # Flux along trench segment
            volume_rate = thickness * segment_length * subduction_rate
            volume_rate = np.clip(volume_rate, 0.0, np.inf)

            # Flux in each grid cell
            total_volume_rate, _, _ = np.histogram2d(
                x=subset["lon"],
                y=subset["lat"],
                bins=(xedges, yedges),
                weights=volume_rate,
            )
            total_volume_rate = total_volume_rate.T

            # Convert to per-unit-area rate
            density = total_volume_rate / cell_areas

            grids[column].append(density)

    grids = {i: np.dstack(grids[i]) for i in grids}
    cumulative_grids = {
        i: np.cumsum(grids[i], axis=-1)
        for i in grids
    }

    colname_map = {
        "sediment_thickness (m)": "subducted_sediment_equivalent_thickness (m)",
        "plate_thickness (m)": "subducted_plate_equivalent_thickness (m)",
        "carbonate_thickness (m)": "subducted_carbonate_equivalent_thickness (m)",
        "total_water_thickness (m)": "subducted_water_equivalent_thickness (m)",
        "crustal_carbon_density (t/m^2)": "subducted_crustal_carbon_density (t/m^2)",
        "carbonate_carbon_density (t/m^2)": "subducted_carbonate_carbon_density (t/m^2)",
        "total_carbon_density (t/m^2)": "subducted_total_carbon_density (t/m^2)",
    }

    groups = [(t, g) for t, g in data.groupby("age (Ma)", sort=True)]

    def _process_one_time(t, subset):
        to_concat_cols = [subset]

        if plate_reconstruction is not None:
            plate_map = _create_plate_map(
                time=t,
                plate_reconstruction=plate_reconstruction,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
                resolution=grid_resolution,
            )
        else:
            plate_map = None

        # Define the lag window: [t, t + time_lag]
        window_end = t
        window_start = min(t + time_lag, times.max())

        # Find all discrete reconstruction times inside that window
        window_idx = np.where(
            (times >= window_end) & (times <= window_start)
        )[0]

        if window_idx.size == 0:
            raise ValueError(
                f"No reconstruction times found in lag window [{window_start}, {window_end}] for t={t}."
            )

        idx_start = window_idx[0]   # older edge of window
        idx_end = window_idx[-1]    # younger edge of window (= time t)

        for column in columns:
            if time_lag > 0:
                # Sum only within the lag window using cumulative differences.
                # Because times are sorted descending (older -> younger),
                # cumulative_grids[..., idx] contains everything from the oldest
                # time down to idx. To isolate the window, subtract the cumulative
                # value just before the window starts.
                if idx_start == 0:
                    raster = cumulative_grids[column][..., idx_end]
                else:
                    raster = (
                        cumulative_grids[column][..., idx_end]
                        - cumulative_grids[column][..., idx_start - 1]
                    )
            else:
                # Original behaviour: cumulative from the start of the reconstruction
                # down to time t.
                raster = cumulative_grids[column][..., idx_end]

            new_col = _coregister_raster(
                raster=raster,
                points=subset,
                plate_map=plate_map,
                plate_reconstruction=plate_reconstruction,
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
                method=method,
            )

            new_colname = colname_map[column]
            new_col.name = new_colname

            if column == "sediment_thickness (m)":
                v = subset["convergence_rate (cm/yr)"].to_numpy()
                factor = 0.6 + 0.4 * np.clip(v / 6.8, 0.0, 1.0)
                new_col = new_col * factor
                new_col.name = new_colname

            to_concat_cols.append(new_col)

        return pd.concat(to_concat_cols, axis="columns")

    to_concat_rows = Parallel(n_jobs=n_jobs, verbose=int(verbose))(
        delayed(_process_one_time)(t, subset) for t, subset in groups
    )

    return pd.concat(to_concat_rows, axis="index")


def _longitude_length(latitude, delta=1.0, radius=EARTH_RADIUS * 1000.0, degrees=True):
    
    if degrees:
        latitude = np.deg2rad(latitude)
        length = np.deg2rad(1.0) * radius * np.cos(latitude)
    else:
        length = radius * np.cos(latitude)
    
    return delta * length


def _latitude_length(delta=1.0, radius=EARTH_RADIUS * 1000.0, degrees=True):
    
    if degrees:
        delta = np.deg2rad(delta)
        
    return radius * delta


def _coregister_raster(
    raster,
    points: pd.DataFrame,
    plate_map: Optional[Raster] = None,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[pygplates.RotationModel] = None,
    topology_features: Optional[pygplates.FeatureCollection] = None,
    static_polygons: Optional[pygplates.FeatureCollection] = None,
    anchor_plate_id: int = None,
    time: Optional[float] = None,
    method="nearest",
):
    
    raster = Raster(raster)
    if plate_map is None and plate_reconstruction is not None and time is not None:
        plate_map = _create_plate_map(
            time=time,
            plate_reconstruction=plate_reconstruction,
            rotation_model=rotation_model,
            topology_features=topology_features,
            static_polygons=static_polygons,
            anchor_plate_id=anchor_plate_id,
            resolution=360 / (raster.shape[1] - 1),
        ).data
    elif plate_map is not None:
        plate_map = np.array(plate_map)

    if plate_map is None or "subducting_plate_ID" not in points.columns:
        raster = raster.fill_NaNs()
        new_col = raster.interpolate(
            lons=points["lon"],
            lats=points["lat"],
            method=method,
        )
        new_col = pd.Series(new_col, index=points.index)
    else:
        raster = raster.resize(plate_map.shape[1], plate_map.shape[0])
        new_col = pd.Series(np.nan, index=points.index)
        for plate_id, subset_pid in points.groupby("subducting_plate_ID"):
            arr_tmp = np.array(raster)
            arr_tmp[plate_map != plate_id] = np.nan
            raster_pid = Raster(arr_tmp).fill_NaNs()
            intpd = raster_pid.interpolate(
                lons=subset_pid["lon"],
                lats=subset_pid["lat"],
                method=method,
            )
            for i, val in zip(subset_pid.index, intpd):
                new_col.at[i] = val
                
    return new_col


def format_feature_name(s, bold=False):
    
    """Make feature names easier to read in plots."""
    s = s.replace("_", " ")
    s = s[0].capitalize() + s[1:]

    replace = {
        "(cm/yr)": r"($\mathrm{cm \; {yr}^{-1}}$)",
        "(m)": r"($\mathrm{m}$)",
        "(m^3/m^2)": r"($\mathrm{m^3 \; m^{-2}}$)",
        "(m^2/yr)": r"($\mathrm{m^2 \; {yr}^{-1}}$)",
        "(t/m^2)": r"($\mathrm{t \; m^{-2}}$)",
        "(Ma)": r"($\mathrm{Ma}$)",
        "(degrees)": r"($\mathrm{\degree}$)",
        "(km)": r"($\mathrm{km}$)",
        "(km/Myr)": r"($\mathrm{km \; {Myr}^{-1}}$)",
        "(/Ps)": r"($\mathrm{{Ps}^{-1}}$)",
        "(/s)": r"($\mathrm{{s}^{-1}}$)",
        "(rad/Ps)": r"($\mathrm{rad. \; {Ps}^{-1}}$)",
    }
    if bold:
        replace = {
            key: value.replace(r"\mathrm", r"\mathbf")
            for key, value in replace.items()
        }
    for key, value in replace.items():
        s = s.replace(key, value)

    return s


def clean_feature_name(name):
    # remove anything in parentheses (including the parentheses)
    name = re.sub(r"\s*\(.*?\)", "", name)
    return name.strip()


def run_calculate_outflux(
        input_data,
        times,
        water_components,
        water_headers,
        carbon_components,
        carbon_headers,
        lookup_tables,
        lookup_interp,
        agegrid_dir,
        sedthick_dir,
        lithosphere_top_dir,
        input_water_cdf_filename,
        input_carbon_cdf_filename,
        output_filename=None,
        n_jobs=-2,
        verbose=False,
        ):

    if isinstance(input_data, str):
        input_data = pd.read_csv(input_data)
    if isinstance(input_data, pd.DataFrame):
        input_data = [
            (input_data[input_data["age (Ma)"] == time]).copy()
            for time in times
        ]

    times_split = np.array_split(times, n_jobs)
    df_array = np.empty(len(input_data), dtype="object")
    for i, df in enumerate(input_data):
        df_array[i] = df
    input_data_split = np.array_split(df_array, n_jobs)

    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        results = parallel(
            delayed(_calculate_outflux_subset)(
                times=t,
                dfs=d,
                water_components=water_components,
                water_headers=water_headers,
                carbon_components=carbon_components,
                carbon_headers=carbon_headers,
                lookup_tables=lookup_tables,
                lookup_interp=lookup_interp,
                agegrid_dir=agegrid_dir,
                sedthick_dir=sedthick_dir,
                lithosphere_top_dir=lithosphere_top_dir,
                input_water_cdf_filename=input_water_cdf_filename,
                input_carbon_cdf_filename=input_carbon_cdf_filename,
            )
            for t, d in zip(times_split, input_data_split)
        )
    out = []
    for i in results:
        out.extend(i)

    out = pd.concat(out, ignore_index=True)
    
    # Save to CSV if a path is provided
    if output_filename:
        out.to_csv(output_filename, index=False)
        if verbose:
            print(f"Results written to: {output_filename}")
        
    return out


def _calculate_outflux_subset(
        times,
        dfs,
        water_components,
        water_headers,
        carbon_components,
        carbon_headers,
        lookup_tables,
        lookup_interp,
        agegrid_dir,
        sedthick_dir,
        lithosphere_top_dir,
        input_water_cdf_filename,
        input_carbon_cdf_filename,
        ):
    
    return [
        _calculate_outflux(
            t,
            df,
            water_components,
            water_headers,
            carbon_components,
            carbon_headers,
            lookup_tables,
            lookup_interp,
            agegrid_dir,
            sedthick_dir,
            lithosphere_top_dir,
            input_water_cdf_filename,
            input_carbon_cdf_filename,
            )
        for t, df in zip(times, dfs)
    ]


def _calculate_outflux(
        time,
        df_time,
        water_components,
        water_headers,
        carbon_components,
        carbon_headers,
        lookup_tables,
        lookup_interp,
        agegrid_dir,
        sedthick_dir,
        lithosphere_top_dir,
        input_water_cdf_filename,
        input_carbon_cdf_filename,
        ):
    
    # reconstruct water components
    df_time = _reconstruct_slab_water_storage(
        time,
        df_time,
        water_components,
        water_headers,
        lookup_tables,
        lookup_interp,
        agegrid_dir,
        sedthick_dir,
        lithosphere_top_dir,
        input_water_cdf_filename,
        )
    
    df_time = _reconstruct_melting_rate(df_time)
    
    # reconstruct carbon components
    df_time = _reconstruct_slab_carbon_storage(
        time,
        df_time,
        carbon_components,
        carbon_headers,
        lookup_tables,
        lookup_interp,
        sedthick_dir,
        agegrid_dir,
        input_carbon_cdf_filename,
        )
    
    df_time = _reconstruct_slab_carbon_outflux(df_time)
    
    # change units from Mt/m2 to t/m/yr
    cols = ['subduction_water_flux_lithosphere (t/m/yr)',
            'slab_water_outflux_lithosphere (t/m/yr)',
            'subduction_water_flux_crust (t/m/yr)',
            'slab_water_outflux_crust (t/m/yr)',
            'subduction_water_flux_sediment (t/m/yr)',
            'slab_water_outflux_sediment (t/m/yr)',
            'total_slab_water_outflux (t/m/yr)',
            'melting_rate (t/m/yr)',
            'subduction_carbon_flux_lithosphere (t/m/yr)',
            'slab_carbon_outflux_lithosphere (t/m/yr)',
            'subduction_carbon_flux_crust (t/m/yr)',
            'slab_carbon_outflux_crust (t/m/yr)',
            'subduction_carbon_flux_serpentinite (t/m/yr)',
            'slab_carbon_outflux_serpentinite (t/m/yr)',
            'subduction_carbon_flux_sediment (t/m/yr)',
            'slab_carbon_outflux_sediment (t/m/yr)',
            'subduction_carbon_flux_organic_sediments (t/m/yr)',
            'slab_carbon_outflux_organic_sediments (t/m/yr)',
            'total_slab_carbon_outflux (t/m/yr)']
    
    for col in cols:
        df_time.loc[:,col] = df_time.loc[:,col] * np.clip(df_time['convergence_rate_orthogonal (cm/yr)'], 0, 1e99) * 1e6
        
    # df_time[cols] = df_time[cols].clip(lower=0)
    
    return df_time


def _reconstruct_slab_water_storage(
        time,
        df_time,
        water_components,
        water_headers,
        lookup_tables,
        lookup_interp,
        agegrid_dir,
        sedthick_dir,
        lithosphere_top_dir,
        input_water_cdf_filename,
        ):
    
    # remove entries that have "negative" subduction
    # this occurs when the subduction obliquity is greater than 90 degrees
    subduction_convergence = np.clip(df_time['convergence_rate_orthogonal (cm/yr)'], 0, 1e99)

    # sample AgeGrid
    agegrid_filename = os.path.join(agegrid_dir, f"seafloor_age_{time:0.0f}Ma.nc")
    age_grid = Raster(filename=agegrid_filename)
    age_grid.fill_NaNs(inplace=True)
    age_interp = age_grid.interpolate(df_time['lon'], df_time['lat'])
    thickness = plate_isotherm_depth(age_interp, n=50, tol=1)
    thickness_brittle = plate_isotherm_depth(age_interp, temp=600, n=50, tol=1)
    
    subduction_dip = np.clip(df_time['slab_dip (degrees)'], 1, 90)
    
    def wt_bending_serpentinite(dip):
        
        return np.clip(dip, 0, 90)*1.0/9
    
    # serpentinite due to slab dip
    wt_serp = wt_bending_serpentinite(subduction_dip)/100

    # kg/m2 = thickness * H2O wt % * serpentinite wt % * density
    water_dip = (thickness_brittle - 8e3)*(0.13*wt_serp)*2900
    water_dip *= 1e-9 # convert kg/m2 to Mt/m2
    
    # get sediment thickness
    sediments_filename = os.path.join(sedthick_dir, f"sediment_thickness_{time:0.0f}Ma.nc")
    seds_grid = Raster(filename=sediments_filename.format(time))
    seds_grid.fill_NaNs(inplace=True)
    sediment_thickness_interp = seds_grid.interpolate(df_time['lon'], df_time['lat'])
    
    ##
    ## Calculate P-T conditions of slab
    ##

    Phi = _calcPhi(age_interp, subduction_convergence*100, subduction_dip)
    T_sediments, T_volcanics, T_moho = _calcSlabTemperatures(Phi+1e-12)

    # Calculations are for a constant pressure (4.3750 GPa) at 125 km
    # Assumed 9.81 m/s (gravity) and density of 3500 kg/m3
    P = np.full_like(T_sediments, 125e3*9.81*3500e3*1e-8)

    # convert H2O wt % to kg/m3 (multiply by constant density)
    H2O_sediments_bound = _sample_lookup_table(P, T_sediments+273.14, 0, lookup_tables, lookup_interp) * 1700.0/100 # density
    H2O_crust_bound     = _sample_lookup_table(P, T_volcanics+273.14, 1, lookup_tables, lookup_interp) * 3000.0/100 # density
    H2O_mantlelit       = _sample_lookup_table(P, T_moho+273.14, 3, lookup_tables, lookup_interp) * 3500.0/100

    # integrate by thickness (now kg/m2 H2O)
    H2O_sediments_bound *= sediment_thickness_interp # 100 m thickness
    H2O_crust_bound     *= 7e3 # 7 km thickness
    H2O_mantlelit       *= (thickness-7e3) # lithospheric thickness - 7e3

    # convert kg/m2 to Mt/m2 water
    H2O_sediments_bound *= 1e-9
    H2O_crust_bound     *= 1e-9
    H2O_mantlelit       *= 1e-9
    
    H2O_component = [H2O_mantlelit, H2O_crust_bound, H2O_sediments_bound]
    
    for c, component in enumerate(water_components):
        water_type = water_headers[c].split('_')[0]
        
        # read grids in units of Mt/m2
        water_grid = Raster(filename=input_water_cdf_filename.format(component, "mean", water_type, time))
        water_grid.fill_NaNs(inplace=True)
        
        if water_type == 'lithosphere':
            # add on lithosphere_top component
            water_grid2 = Raster(filename=input_water_cdf_filename.format(
                lithosphere_top_dir, "mean", water_type, time))
            water_grid2.fill_NaNs(inplace=True)
            water_grid.data += water_grid2.data
                                            
        # interpolate to trenches
        water_interp, (ci, cj) = water_grid.interpolate(df_time['lon'], df_time['lat'], return_indices=True)

        if water_type == "lithosphere":
            # add on water from plate bending
            water_interp += water_dip # Mt/m2

        # remaining water is anything left in the slab that hasn't been devolatilised
        water_remaining = np.minimum(H2O_component[c], water_interp)
        water_outflux = np.clip(water_interp - water_remaining, 0.0, 1e99)
    
        col_subflux = "subduction_water_flux_{} (t/m/yr)".format(water_type)
        col_outflux = "slab_water_outflux_{} (t/m/yr)".format(water_type)
        df_time = df_time.assign(**{col_subflux:water_interp, col_outflux:water_outflux})
        
    return df_time


def _calcPhi(slabAge, slabVelocity, slabDip):
    """
    Take in values of slab dip and return phi, the thermal parameter.
    
    Parameters
    ----------
    slabAge : age of slab (Ma)
    slabVelocity : convergence rate of slab (cm/yr)
    slabDip : dip angle of slab from trench hinge (degrees)
    
    Returns
    -------
    Phi : the thermal parameter
    """
    
    # phiTP = A(ge) * V(elocity) * sin(\delta)
    # Age: Myrs -> yrs
    slabAge = slabAge * 1e+6
    # Slab velocity: cm/year -> km/yr
    slabVelocity = slabVelocity*1e-5
    # Slab dip calculation, note numpy does things in radians, need to convert to degrees
    slabDip = np.sin(np.deg2rad(slabDip))
    
    # Return the product phiTP
    return slabAge*slabVelocity*slabDip


def _calcSlabTemperatures(Phi):
    """
    Calculate temperature, equations from Phi
    
    Returns
    -------
    T_sediments : temperature (Celsius) at sediments interface
    T_volcanics : temperature (Celsius) at volcanics interface
    T_Moho      : temperature (Celsius) at the Moho
    """
    
    T_Sediments = 1331. - 58.6*np.log(Phi)
    T_Volcanics = 1303. - 60.23*np.log(Phi)
    T_Moho      = 1622. - 132.5*np.log(Phi)
        
    return T_Sediments, T_Volcanics, T_Moho


def _sample_lookup_table(P, T, lithology, lookup_tables, lookup_interp):
    """
    Interpolation function. Takes in the approximated Depth to pressure conversion and
    calculated temperature from the slab tops using Van Keken (2011) and the thermal parameter.
    
                                                 0   1    2
    Read in H2O data generated from Perple_X.  [ T,  P,  H2O]
    
    Parameters
    ----------
    P : pressure in bars
    T : temperature in Kelvin
    lithology : lithology ID
        0 = sediments
        1 = metabasalts
        2 = intrusives
        3 = sublithospheric oceanic mantle
        
    Returns
    -------
    H2O : H2O wt. % for a given pressure and temperature from the specified lookup table lithology
    """

    lookup_interp.values = lookup_tables[lithology]
    
    return lookup_interp((P,T))


def _reconstruct_melting_rate(
        df_time,
        d_tol=400,
        rho_m = 3300, # mantle density - kg/m3
        d_arc = 125e3, # arc depth - m
        P_range = [2.5, 3, 3.5, 4, 4.5], # pressure range - GPa
        T_m = 1350, # mantle temperature - Celsius
        ):
    
    # load subduction data
    slab_outflux_area = _calc_total_slab_outflux(df_time)

    # convert Mt/m2 -> wt % in the upper 125 km of the asthenospheric mantle (arc depth and above).
    H2O_wt = 100*slab_outflux_area*1e6/(rho_m * d_arc * 1e-3)

    # estimate melt fraction
    F = np.zeros(H2O_wt.size)
    for P in P_range:
        F += mlt.F_wet(P, T_m, H2O_wt)
    F /= len(P_range)

    # calculate melting rate (melt fraction is in wt % so we convert to kg/m2)
    melting_rate = (F/100)*rho_m*d_arc
    melting_rate *= 1e-9 # convert to Mt/m2

    df_time = df_time.assign(**{"total_slab_water_outflux (t/m/yr)":slab_outflux_area, "melting_rate (t/m/yr)":melting_rate})
    
    return df_time


def _calc_total_slab_outflux(df):
    """ Calculate the slab outflux
    
    Arguments
    ---------
        df : pandas DataFrame
        
    Returns
    -------
        outflux : array
    """
    
    total_slab_outflux = (
        df['slab_water_outflux_lithosphere (t/m/yr)'] + 
        df['slab_water_outflux_crust (t/m/yr)'] + 
        df['slab_water_outflux_sediment (t/m/yr)']
    )
    
    return total_slab_outflux


def _reconstruct_slab_carbon_storage(
        time,
        df_time,
        carbon_components,
        carbon_headers,
        lookup_tables,
        lookup_interp,
        sedthick_dir,
        agegrid_dir,
        input_carbon_cdf_filename,
        spacingX=0.1,
        spacingY=0.1,
        ):
        
    sediments_filename = os.path.join(sedthick_dir, f"sediment_thickness_{time:0.0f}Ma.nc")
    sediment_thickness_grid = gplately.grids.read_netcdf_grid(sediments_filename.format(time),
                                                              resample=(spacingX,spacingY))
    sediment_thickness_filled = gplately.grids.fill_raster(sediment_thickness_grid)
    sediment_thickness_interp = gplately.grids.sample_grid(
        df_time['lon'], df_time['lat'], sediment_thickness_filled)
      
    # remove entries that have "negative" subduction
    # this occurs when the subduction obliquity is greater than 90 degrees
    subduction_convergence = np.clip(df_time['convergence_rate_orthogonal (cm/yr)'], 0, 1e99)

    # sample AgeGrid
    agegrid_filename = os.path.join(agegrid_dir, f"seafloor_age_{time:0.0f}Ma.nc")
    age_grid = Raster(filename=agegrid_filename.format(time))
    age_grid.fill_NaNs(inplace=True)
    age_interp = age_grid.interpolate(df_time['lon'], df_time['lat'])
    thickness = gplately.tools.plate_isotherm_depth(age_interp, n=50, tol=1)
    
    subduction_dip = np.clip(df_time['slab_dip (degrees)'], 1, 90)

    Phi = _calcPhi(age_interp, subduction_convergence*100, subduction_dip)
    T_sediments, T_volcanics, T_moho = _calcSlabTemperatures(Phi+1e-12)

    # Calculations are for a constant pressure (4.3750 GPa) at 125 km
    # Assumed 9.81 m/s (gravity) and density of 3500 kg/m3
    P = np.full_like(T_sediments, 125e3*9.81*3500e3*1e-8)

    # convert CO2 wt % to kg/m3 (multiply by constant density)
    CO2_sediments = _sample_lookup_table(P, T_sediments+273.14, 4+0, lookup_tables, lookup_interp) * 3500.0/100 # density
    CO2_volcanics = _sample_lookup_table(P, T_volcanics+273.14, 4+1, lookup_tables, lookup_interp) * 3500.0/100 # density
    CO2_intrusive = _sample_lookup_table(P, T_moho+273.14, 4+3, lookup_tables, lookup_interp) * 3500.0/100 # density
    CO2_mantlelit = _sample_lookup_table(P, T_moho+273.14, 4+3, lookup_tables, lookup_interp) * 3500.0/100
    
    # correct CO2 volcanics by total amount that is possible to extract
    # CO2_volcanics = np.minimum(CO2_volcanics, crust_wtpercent_interp * 3500.0/100)

    # integrate by thickness and rate (now kg CO2/m2)
    CO2_sediments *= sediment_thickness_interp # 100m thickness
    CO2_volcanics *= 450.0 # 450m thickness
    CO2_intrusive *= 30e3*0.4  # 7km thickness
    CO2_mantlelit *= (thickness-sediment_thickness_interp-7e3)

    # convert kg CO2/m2 to Mt C/m2 carbon from ratio of molecular weight
    carbon_sediments = CO2_sediments * 12.0107/44.0095 * 1e-9
    carbon_volcanics = CO2_volcanics * 12.0107/44.0095 * 1e-9
    carbon_intrusive = CO2_intrusive * 12.0107/44.0095 * 1e-9
    carbon_mantlelit = CO2_mantlelit * 12.0107/44.0095 * 1e-9


    carbon_component = [carbon_mantlelit, carbon_intrusive, carbon_volcanics, carbon_sediments, carbon_sediments]

    for c, component in enumerate(carbon_components):
        carbon_type = carbon_headers[c]
        
        # carbon grids in units of Mt/m2
        carbon_grid = Raster(filename=input_carbon_cdf_filename.format(
                                        component, "mean", carbon_type, time))
        carbon_grid.fill_NaNs(inplace=True)
                                            
        # interpolate to trenches
        carbon_interp, (ci, cj) = carbon_grid.interpolate(df_time['lon'], df_time['lat'], return_indices=True)

        # remaining water is anything left in the slab that hasn't been devolatilised
        carbon_remaining = np.minimum(carbon_component[c], carbon_interp)
        carbon_outflux = np.clip(carbon_interp - carbon_remaining, 0.0, 1e99)

        col_subflux = "subduction_carbon_flux_{} (t/m/yr)".format(carbon_type)
        col_outflux = "slab_carbon_outflux_{} (t/m/yr)".format(carbon_type)
        df_time = df_time.assign(**{col_subflux:carbon_interp, col_outflux:carbon_outflux})
        
    return df_time


def _reconstruct_slab_carbon_outflux(df):
    """ Calculate the slab outflux
    
    Arguments
    ---------
        df : pandas DataFrame
        
    Returns
    -------
        outflux : array
    """
    total_slab_outflux = (
        df['slab_carbon_outflux_lithosphere (t/m/yr)'] + 
        df['slab_carbon_outflux_crust (t/m/yr)'] + 
        df['slab_carbon_outflux_sediment (t/m/yr)'] + 
        df['slab_carbon_outflux_organic_sediments (t/m/yr)'] +
        df['slab_carbon_outflux_serpentinite (t/m/yr)']
    )

    df = df.assign(**{'total_slab_carbon_outflux (t/m/yr)':total_slab_outflux})
    
    return df