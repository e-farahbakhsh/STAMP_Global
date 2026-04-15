'''
Feature Extraction

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 16/09/2025
'''


import os
from sys import stderr
from typing import (
    Iterable,
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
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
import xarray as xr

from gplately import (
    EARTH_RADIUS,
    PlateReconstruction,
    PlotTopologies,
    Raster,
)
from gplately.tools import plate_isotherm_depth


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


def run_calculate_convergence(
    min_time: float,
    max_time: float,
    temporal_resolution: int,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    rotation_model: Optional[Union[Sequence[str], str]] = None,
    topology_features: Optional[Sequence[str]] = None,
    static_polygons: Optional[Sequence[str]] = None,
    anchor_plate_id: int = 0,
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
    rotation_model : str or sequence of str, optional
        Rotation model file(s) describing plate motions.
    topology_features : sequence of str, optional
        Topological feature collection(s) (e.g., plate boundaries).
    static_polygons : sequence of str, optional
        Static polygon file(s) used for plate partitioning.
    plate_reconstruction : PlateReconstruction, optional
        Pre-constructed PlateReconstruction object. If not provided,
        one will be created internally from the input files.
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
    anchor_plate_id: int = 0,
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
    anchor_plate_id: int = 0,
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

    if spreadrate_dir is None and agegrid_dir is not None:
        spreadrate_dir = agegrid_dir

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
    anchor_plate_id: int = 0,
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
    anchor_plate_id: int = 0,
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
    anchor_plate_id: int = 0,
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

    df["seafloor_age (Ma)"] = np.nan
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
        "agegrid": "seafloor_age (Ma)",
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
    anchor_plate_id: int = 0,
    resolution: float = 0.1, # degrees
    tessellate_degrees: Optional[float] = None,
    output_filename: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = False,
) -> Raster:
    
    time = float(time)
    resolution = float(resolution)
    if tessellate_degrees is None:
        tessellate_degrees = resolution
    tessellate_degrees = float(tessellate_degrees)

    if not isinstance(plate_reconstruction, PlateReconstruction):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topology_features,
                static_polygons=static_polygons,
                anchor_plate_id=anchor_plate_id,
            )
                
    gplot = PlotTopologies(plate_reconstruction, anchor_plate_id=anchor_plate_id)
    gplot.time = time

    topologies = gplot.get_all_topologies(
        tessellate_degrees=tessellate_degrees,
    )
    topologies["feature_type"] = topologies["feature_type"].astype(str)

    sort_key = lambda ftype: ftype.apply(
        lambda s: {
            "gpml:TopologicalNetwork": 0,
            "gpml:OceanicCrust": 1,
            "gpml:TopologicalClosedPlateBoundary": 2,
        }.get(s, -1)
    )
    minx = -180
    maxx = 180
    miny = -90
    maxy = 90

    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)
    nx = lons.size
    ny = lats.size
    transform = from_bounds(
        minx,
        miny,
        maxx,
        maxy,
        nx,
        ny,
    )

    topologies = topologies.sort_values(by="feature_type", key=sort_key)
    shapes = zip(
        topologies["geometry"],
        topologies["reconstruction_plate_ID"],
    )
    grid = rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        fill=-1,
        dtype=np.int_,
        merge_alg=MergeAlg.replace,
        transform=transform,
    )
    # Output is always upper-left origin
    grid = np.flipud(grid)  # Convert to lower-left
    raster = Raster(grid, extent="global", origin="lower")
    if output_filename is not None:
        if verbose:
            print(
                " - Writing output file: "
                + os.path.basename(output_filename),
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
        ages = np.array(df["seafloor_age (Ma)"])
        thicknesses = plate_isotherm_depth(ages, maxiter=100)
    slab_flux = rates * thicknesses
    df["slab_flux (m^2/yr)"] = slab_flux
    
    return df


def extract_subducted_thickness(
    data,
    columns=None,
    grid_resolution=0.5,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    anchor_plate_id: int = 0,
    method="nearest",
):
    
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
    grid_resolution : float, default=0.5
        Spatial resolution of the output grid in degrees.
    plate_reconstruction : PlateReconstruction, optional
        If provided, generates plate ID rasters to ensure subducted
        volumes are correctly partitioned by plate.
    method : str, default="nearest"
        Interpolation method for mapping gridded volumes back to trench
        points. Options include `nearest` or `linear` depending on
        the Raster class.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with additional columns giving cumulative
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

    * Results are accumulated over time (`np.cumsum`) to track the
      integrated subducted volumes.
    * Grid cell areas are computed using spherical geometry so that
      volume fluxes are normalised by surface area before interpolation.
    """

    if columns == "default" or columns is None:
        columns = [
            i for i in
            [
                "sediment_thickness (m)",
                "plate_thickness (m)",
                "carbonate_thickness (m)",
                "total_water_thickness (m)",
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
            # Thickness in m
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
            # Volume of material subducted along trench segment in m^3/Myr
            volume_rate = thickness * segment_length * subduction_rate
            volume_rate = np.clip(volume_rate, 0.0, np.inf)

            # Volume subducted in each grid cell in m^3/Myr
            total_volume_rate, _, _ = np.histogram2d(
                x=subset["lon"],
                y=subset["lat"],
                bins=(xedges, yedges),
                weights=volume_rate,
            )
            total_volume_rate = total_volume_rate.T

            # Volume subducted per unit area in m/Myr (m^3/Myr / m)
            density = total_volume_rate / cell_areas

            grids[column].append(density)

    grids = {i: np.dstack(grids[i]) for i in grids}
    cumulative_grids = {
        i: np.cumsum(grids[i], axis=-1)
        for i in grids
    }

    colname_map = {
        "sediment_thickness (m)": "subducted_sediment_volume (m)",
        "plate_thickness (m)": "subducted_plate_volume (m)",
        "carbonate_thickness (m)": "subducted_carbonate_volume (m)",
        "total_water_thickness (m)": "subducted_water_volume (m)",
        "total_carbon_density (t/m^2)": "subducted_carbon_density (t/m^2)",
    }

    to_concat_rows = []
    for time, subset in data.groupby("age (Ma)"):
        to_concat_cols = [subset]
        idx = np.where(times == time)[0][0]
        if plate_reconstruction is not None:
            plate_map = _create_plate_map(
                time=time,
                plate_reconstruction=plate_reconstruction,
                anchor_plate_id=anchor_plate_id,
                resolution=grid_resolution,
            )
        else:
            plate_map = None
        for column in columns:
            raster = cumulative_grids[column][..., idx]
            new_col = _coregister_raster(
                raster=raster,
                points=subset,
                plate_map=plate_map,
                method=method,
            )
            new_colname = colname_map.get(
                column,
                (
                    "subducted_"
                    + column.split()[0].replace('_thickness', '_volume')
                    + " (m)"
                )
            )
            new_col.name = new_colname
            to_concat_cols.append(new_col)
        to_concat_rows.append(pd.concat(to_concat_cols, axis="columns"))
        
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
    anchor_plate_id: int = 0,
    time: Optional[float] = None,
    method="nearest",
):
    
    raster = Raster(raster)
    if plate_map is None and plate_reconstruction is not None and time is not None:
        plate_map = _create_plate_map(
            time=time,
            plate_reconstruction=plate_reconstruction,
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
