import concurrent.futures
import glob
from multiprocessing import cpu_count
import os
from sys import stderr
from tempfile import NamedTemporaryFile
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
from gplately.reconstruction import reconstruct_points
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import rioxarray as rio
import seaborn as sns
from shapely.geometry import MultiPoint, MultiPolygon, Point
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.ops import linemerge
from skimage.transform import resize
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import xarray as xr

from plate_model_manager import PlateModelManager
import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    Raster,
    EARTH_RADIUS,
)
from gplately.geometry import (
    pygplates_to_shapely,
    wrap_geometries,
)
from gplately.tools import (
    plate_isotherm_depth,
    xyz2lonlat,
)


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

DEFAULT_RESOLUTION = 0.1 # degrees
DEFAULT_DISTANCE_THRESHOLD = 0.1  # degrees

PMM = PlateModelManager()


def run_calculate_convergence(
    nprocs: int,
    min_time: float,
    max_time: float,
    temporal_resolution: int,
    topology_filenames: Optional[Sequence[str]] = None,
    rotation_filenames: Optional[Union[Sequence[str], str]] = None,
    output_dir: _PathLike = ".",
    verbose: bool = False,
    plate_reconstruction: Optional[PlateReconstruction] = None,
):

    if not os.path.exists(output_dir):
        if verbose:
            print(
                "Output directory does not exist; creating now: "
                + str(output_dir),
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    if plate_reconstruction is None:
        if topology_filenames is None or rotation_filenames is None:
            raise TypeError(
                "Either `topology_filenames` and `rotation_filenames` "
                "or `plate_reconstruction` must be specified."
            )
        plate_reconstruction = PlateReconstruction(
            rotation_model=rotation_filenames,
            topology_features=topology_filenames,
        )

    times = np.arange(min_time, max_time + temporal_resolution, temporal_resolution)
    
    if nprocs == 1:
        data = [
            _parallel_func(
                plate_reconstruction=plate_reconstruction,
                time=t,
                ignore_warnings=True,
            )
            for t in times
        ]
    else:
        from joblib import Parallel, delayed

        with Parallel(nprocs, verbose=10 if verbose else 0) as parallel:
            data = parallel(
                delayed(_parallel_func)(
                    plate_reconstruction=plate_reconstruction,
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
        if col not in data.columns:
            continue
        x_km = np.deg2rad(data[col]) * EARTH_RADIUS
        data[col.replace("(degrees)", "(km)")] = x_km
        data = data.drop(columns=col, errors="ignore")
        
    return data


def _parallel_func(
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
    out["age (Ma)"] = np.float_(time)
    
    return out

# --------------------------------------------------

def get_plate_reconstruction(
    model_name: Optional[str] = None,
    model_dir: str = "plate_model",
    anchor_plate_id: int = 0,
    filter_topologies: bool = False,
):

    if model_name is None:
        globs = ["*.gpml", "*.gpmlz"]
        rotation_files = []
        topology_files = []
        static_polygons = []
        for g in globs:
            all_filenames = glob.glob(os.path.join(model_dir, "**", g), recursive=True)
            topology_files.extend(glob.glob(os.path.join(model_dir, g)))
            static_polygons.extend(
                [
                    i for i in all_filenames
                    if "static" in os.path.basename(i).lower()
                    and "polygon" in os.path.basename(i).lower()
                ]
            )
        rotation_files.extend(
            glob.glob(os.path.join(model_dir, "**", "*.rot"), recursive=True)
        )

    else:
        model = _fetch(model_name, os.path.join(model_dir, ".downloaded"))
        rotation_files = model.get_rotation_model()
        topology_files = model.get_layer("Topologies")
        static_polygons = model.get_layer("StaticPolygons")

    if filter_topologies:
        topology_features = filter_topological_features(topology_files)
        tf = NamedTemporaryFile(suffix=".gpml", delete=False)
        tf.close()
        topology_features.write(tf.name)
        topology_files = [tf.name]

    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_files,
        topology_features=topology_files,
        static_polygons=static_polygons,
        anchor_plate_id=anchor_plate_id,
    )
    
    if filter_topologies:
        return plate_reconstruction, tf
    return plate_reconstruction


def _fetch(model_name: str, model_dir: str = "plate_model"):
    
    if model_name not in PMM.get_available_model_names():
        raise ValueError(
            f"Invalid plate model name: {model_name}"
        )
    model = PMM.get_model(
        model_name,
        data_dir=model_dir,
    )
    if model is None:
        raise ValueError(
            f"Invalid plate model name: {model_name}"
        )
        
    return model


def filter_topological_features(
    filenames: Union[str, Iterable[str]]
) -> pygplates.FeatureCollection:
    
    topological_features = []
    for fc, filename in pygplates.FeaturesFunctionArgument(
        filenames
    ).get_files():
        if os.path.basename(filename).lower().startswith("inactive"):
            to_add = [
                feature for feature in fc
                if feature.get_feature_type().to_qualified_string()
                != "gpml:TopologicalNetwork"
            ]
        else:
            to_add = list(fc)
        to_add = [
            feature for feature in to_add
            if feature.get_feature_type().to_qualified_string()
            != "gpml:TopologicalSlabBoundary"
        ]
        topological_features.extend(to_add)
        
    return pygplates.FeatureCollection(topological_features)

# --------------------------------------------------

def run_coregister_ocean_rasters(
    nprocs: int,
    times: Sequence[float],
    input_data: Union[_PathLike, Sequence[pd.DataFrame]],
    output_dir: Optional[_PathLike] = None,
    combined_filename: Optional[_PathLike] = None,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    plates_dir: Optional[_PathLike] = None,
    agegrid_dir: Optional[_PathLike] = None,
    spreadrate_dir: Optional[_PathLike] = None,
    sedthick_dir: Optional[_PathLike] = None,
    carbonate_dir: Optional[_PathLike] = None,
    co2_dir: Optional[_PathLike] = None,
    verbose: bool = False,
) -> pd.DataFrame:

    if isinstance(input_data, str):
        if os.path.isdir(input_data):
            input_data = [
                pd.read_csv(
                    os.path.join(
                        input_data,
                        "convergence_{:.2f}.csv".format(time),
                    )
                )
                for time in times
            ]
        else:
            input_data = pd.read_csv(input_data)
    if isinstance(input_data, pd.DataFrame):
        input_data = [
            (input_data[input_data["age (Ma)"] == time]).copy()
            for time in times
        ]

    if output_dir is not None and not os.path.isdir(output_dir):
        if verbose:
            print(
                "Output directory does not exist; creating now: "
                + str(output_dir),
                file=stderr,
            )
        os.makedirs(output_dir, exist_ok=True)

    if spreadrate_dir is None and agegrid_dir is not None:
        spreadrate_dir = agegrid_dir

    if nprocs == 1:
        out = _run_subset(
            times=times,
            dfs=input_data,
            agegrid_dir=agegrid_dir,
            spreadrate_dir=spreadrate_dir,
            sedthick_dir=sedthick_dir,
            carbonate_dir=carbonate_dir,
            co2_dir=co2_dir,
            output_dir=output_dir,
            plate_reconstruction=plate_reconstruction,
            topology_features=topology_features,
            rotation_model=rotation_model,
            plates_dir=plates_dir,
        )
    else:
        from joblib import Parallel, delayed

        times_split = np.array_split(times, nprocs)
        df_array = np.empty(len(input_data), dtype="object")
        for i, df in enumerate(input_data):
            df_array[i] = df
        input_data_split = np.array_split(df_array, nprocs)

        with Parallel(nprocs, verbose=int(verbose)) as parallel:
            results = parallel(
                delayed(_run_subset)(
                    times=t,
                    dfs=d,
                    agegrid_dir=agegrid_dir,
                    spreadrate_dir=spreadrate_dir,
                    sedthick_dir=sedthick_dir,
                    carbonate_dir=carbonate_dir,
                    co2_dir=co2_dir,
                    output_dir=output_dir,
                    plate_reconstruction=plate_reconstruction,
                    topology_features=topology_features,
                    rotation_model=rotation_model,
                    plates_dir=plates_dir,
                )
                for t, d in zip(times_split, input_data_split)
            )
        out = []
        for i in results:
            out.extend(i)

    out = pd.concat(out, ignore_index=True)

    if combined_filename is not None:
        out.to_csv(combined_filename, index=False)
        
    return out


def _run_subset(
    times,
    dfs,
    agegrid_dir,
    spreadrate_dir,
    sedthick_dir,
    carbonate_dir,
    co2_dir,
    output_dir,
    plate_reconstruction=None,
    topology_features=None,
    rotation_model=None,
    plates_dir=None,
    **kwargs,
):
    
    if plates_dir is None and plate_reconstruction is None:
        if not isinstance(topology_features, pygplates.FeatureCollection):
            topology_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(
                    topology_features
                ).get_features()
            )
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)

    return [
        coregister_ocean_rasters(
            time=t,
            df=df,
            agegrid_dir=agegrid_dir,
            spreadrate_dir=spreadrate_dir,
            sedthick_dir=sedthick_dir,
            carbonate_dir=carbonate_dir,
            co2_dir=co2_dir,
            output_dir=output_dir,
            plate_reconstruction=plate_reconstruction,
            topology_features=topology_features,
            rotation_model=rotation_model,
            plates_dir=plates_dir,
            **kwargs,
        )
        for t, df in zip(times, dfs)
    ]


def coregister_ocean_rasters(
    time: float,
    df: _PathOrDataFrame,
    agegrid_dir: _PathLike,
    spreadrate_dir: _PathLike,
    sedthick_dir: _PathLike,
    carbonate_dir: _PathLike,
    co2_dir: _PathLike,
    output_dir: _PathLike,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topology_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
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
        raster = create_plate_map(
            time=time,
            plate_reconstruction=plate_reconstruction,
            topology_features=topology_features,
            rotation_model=rotation_model,
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

    if output_dir is not None:
        output_filename = os.path.join(
            output_dir, "subduction_data_{}Ma.csv".format(time)
        )
        df.to_csv(output_filename, index=False)

    return df


def create_plate_map(
    time: float,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topology_features: Optional[pygplates.FeatureCollection] = None,
    rotation_model: Optional[pygplates.RotationModel] = None,
    resolution: float = DEFAULT_RESOLUTION,
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
        if (topology_features is None) or (rotation_model is None):
            raise TypeError(
                "Either plate_reconstruction or both of "
                + "topology_features and rotation_model "
                + "must be provided"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topology_features,
            )
    gplot = PlotTopologies(plate_reconstruction)
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

# --------------------------------------------------

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

# --------------------------------------------------

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

# --------------------------------------------------

def extract_subducted_thickness(
    data,
    columns=None,
    grid_resolution=0.5,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    method="nearest",
):

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
            plate_map = create_plate_map(
                time=time,
                plate_reconstruction=plate_reconstruction,
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
    time: Optional[float] = None,
    method="nearest",
):
    
    raster = Raster(raster)
    if plate_map is None and plate_reconstruction is not None and time is not None:
        plate_map = create_plate_map(
            time=time,
            plate_reconstruction=plate_reconstruction,
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

# --------------------------------------------------

def run_create_buffer_zones(
    nprocs: int,
    times: Sequence[float],
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topological_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    output_dir: _PathLike = os.curdir,
    buffer_distance: float = 6,
    verbose: bool = False,
    return_output: bool = False,
) -> Optional[List[gpd.GeoDataFrame]]:

    if plate_reconstruction is None:
        if topological_features is None or rotation_model is None:
            raise TypeError(
                "Either `plate_reconstruction` or both "
                "`topological_features` and `rotation_model` "
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

    times_split = np.array_split(times, nprocs)
    with Parallel(nprocs, verbose=int(verbose)) as parallel:
        results = parallel(
            delayed(_multiple_timesteps_buffer)(
                times=t,
                plate_reconstruction=plate_reconstruction,
                topological_features=topological_features,
                rotation_model=rotation_model,
                output_dir=output_dir,
                buffer_distance=buffer_distance,
                return_output=return_output,
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
    topological_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    output_dir: _PathLike = os.curdir,
):
    
    if plate_reconstruction is None:
        if not isinstance(topological_features, pygplates.FeatureCollection):
            topological_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(
                    topological_features
                ).get_features()
            )
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)

    out = []
    for time in times:
        out.append(
            create_buffer_zones(
                time=time,
                plate_reconstruction=plate_reconstruction,
                topological_features=topological_features,
                rotation_model=rotation_model,
                output_dir=output_dir,
                buffer_distance=buffer_distance,
                return_output=return_output,
            )
        )
        
    if return_output:
        return out


def create_buffer_zones(
    time: float,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    topological_features: Optional[_FeatureCollectionInput] = None,
    rotation_model: Optional[_RotationModelInput] = None,
    output_dir: _PathLike = os.curdir,
    buffer_distance: float = 6,
    clip_to_overriding_plate: bool = False,
    return_output: bool = False,
) -> Optional[gpd.GeoDataFrame]:

    if plate_reconstruction is None:
        if not isinstance(topological_features, pygplates.FeatureCollection):
            topological_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(
                    topological_features
                ).get_features()
            )
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImportWarning)
            plate_reconstruction = PlateReconstruction(
                rotation_model=rotation_model,
                topology_features=topological_features,
            )
    else:
        topological_features = plate_reconstruction.topology_features
        rotation_model = plate_reconstruction.rotation_model

    gplot = PlotTopologies(plate_reconstruction)
    gplot.time = float(time)
    plate_polygons = gplot.get_all_topologies()
    plate_polygons["feature_type"] = plate_polygons["feature_type"].astype(str)
    plate_types = {
        "gpml:TopologicalClosedPlateBoundary",
        "gpml:OceanicCrust",
        "gpml:TopologicalNetwork",
    }
    plate_polygons = plate_polygons[
        plate_polygons["feature_type"].isin(plate_types)
    ]

    topologies = _extract_overriding_plates(
        time=time,
        topological_features=topological_features,
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
        for i in topologies.index:
            if topologies.at[i, "polarity"].lower() != "left":
                topologies.at[i, "geometry"] = topologies.at[i, "geometry"].reverse()
                topologies.at[i, "polarity"] = "Left"
        topologies = _merge_lines(topologies)
        buffered = {}
        for _, row in topologies.iterrows():
            _buffer_sz(row, buffer_distance, topologies.crs, out=buffered)
        buffered = gpd.GeoDataFrame(
            buffered, geometry="geometry", crs=topologies.crs
        )

        if clip_to_overriding_plate:
            clipped = []
            for plate_id in buffered["over"].unique():
                intersection = gpd.overlay(
                    buffered[buffered["over"] == plate_id],
                    plate_polygons[plate_polygons["reconstruction_plate_ID"] == plate_id],
                )
                if len(intersection) > 0:
                    clipped.append(intersection)
            clipped = gpd.GeoDataFrame(pd.concat(clipped, ignore_index=True))
            clipped = clipped[["name", "polarity", "feature_type", "over", "geometry"]]
            clipped = clipped.rename(
                columns={"over": "plate_id", "feature_type": "ftype"}
            )
            buffered = gpd.GeoDataFrame(clipped, geometry="geometry")

    if not buffered.geometry.is_valid.all():
        buffered.geometry = buffered.buffer(0)

    if output_dir is not None:
        output_filename = os.path.join(
            output_dir, f"buffer_zones_{time:0.0f}Ma.geojson"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            buffered.to_file(output_filename)
            
    if return_output:
        return buffered
    
    return None


def _buffer_sz(row, distance_degrees, crs, out):
    
    geom = gpd.GeoSeries(row["geometry"], crs=crs)
    point = geom.representative_point()
    proj = "+proj=aeqd +lat_0={} +lon_0={} +x_0=0 +y_0=0".format(
        point.y, point.x
    )
    projected = geom.to_crs(proj)

    distance_metres = np.deg2rad(distance_degrees) * EARTH_RADIUS * 1000.0
    direction = 1.0 if str(row["polarity"]).lower() == "left" else -1.0
    projected_buffered = projected.buffer(
        distance_metres * direction,
        single_sided=True,
    )
    buffered = projected_buffered.to_crs(crs)
    geometry_out = buffered[0]
    
    # geometries_out = wrap_geometries(
    #     geometry_out, central_meridian=0.0, tessellate_degrees=0.1
    # )
    
    if isinstance(geometry_out, MultiPolygon):
        parts = list(geometry_out.geoms)
    else:
        parts = [geometry_out]
    
    geometries_out = []
    for part in parts:
        wrapped = wrap_geometries(part, central_meridian=0.0, tessellate_degrees=0.1)
        if isinstance(wrapped, (list, tuple)):
            geometries_out.extend(wrapped)
        else:
            geometries_out.append(wrapped)
        
    if isinstance(geometries_out, BaseGeometry):
        geometries_out = [geometries_out]

    for i in geometries_out:
        for column_name in row.index:
            if column_name == "geometry":
                continue
            if column_name not in out:
                out[column_name] = [row[column_name]]
            else:
                out[column_name].append(row[column_name])
        if "geometry" not in out:
            out["geometry"] = [i]
        else:
            out["geometry"].append(i)
            
    return out


def _extract_overriding_plates(
    time,
    topological_features,
    rotation_model,
):
    
    resolved_sections = []
    pygplates.resolve_topologies(
        topological_features,
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

# --------------------------------------------------

def generate_unlabelled_points(
    times,
    input_dir,
    num,
    threads=1,
    output_filename=None,
    seed=None,
    plate_reconstruction=None,
    topological_features=None,
    rotation_model=None,
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
                plate_reconstruction=plate_reconstruction,
                topological_features=topological_features,
                rotation_model=rotation_model,
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
        plate_reconstruction,
        topological_features,
        rotation_model,
):
    
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(seed=rng)

    if plate_reconstruction is None:
        if not isinstance(topological_features, pygplates.FeatureCollection):
            topological_features = pygplates.FeatureCollection(
                pygplates.FeaturesFunctionArgument(
                    topological_features
                ).get_features()
            )
        if not isinstance(rotation_model, pygplates.RotationModel):
            rotation_model = pygplates.RotationModel(rotation_model)

    out = []
    for time in times:
        out.append(
            _generate_points_timestep(
                time=time,
                input_dir=input_dir,
                plate_reconstruction=plate_reconstruction,
                topological_features=topological_features,
                rotation_model=rotation_model,
                num=num,
                rng=rng,
            )
        )
        
    return out


def _generate_points_timestep(
    time,
    input_dir,
    plate_reconstruction,
    topological_features,
    rotation_model,
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
        generated_points = generate_points(
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
        or (topological_features is not None and rotation_model is not None)
    ):
        if time == 0.0:
            present_day_coords = pd.DataFrame(
                {
                    "lon_0": points[:, 0],
                    "lat_0": points[:, 1],
                }
            )
        else:
            present_day_coords = reconstruct_by_topologies(
                data=pd.DataFrame(
                    {
                        "lon": points[:, 0],
                        "lat": points[:, 1],
                        "age (Ma)": time,
                    }
                ),
                plate_reconstruction=plate_reconstruction,
                rotation_model=rotation_model,
                topological_features=topological_features,
                times=np.arange(np.around(time) + 1.0, dtype=np.int_),
                verbose=False,
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


def generate_points(
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

    seed = None

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
        xyz = _generate_points(n=n, rng=rng)
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


def _generate_points(n=1, rng=None):
    
    seed = None

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
    
    seed = None

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


def reconstruct_by_topologies(
    data,
    plate_reconstruction=None,
    rotation_model=None,
    topological_features=None,
    times=None,
    verbose=False,
):

    if plate_reconstruction is not None:
        rotation_model = plate_reconstruction.rotation_model
        topological_features = plate_reconstruction.topology_features
    if rotation_model is None:
        raise TypeError("Rotation model must be provided")
    if topological_features is None:
        raise TypeError("Topological features must be provided")

    data = load_data(data, copy=False)
    if times is None:
        times = np.arange(data["age (Ma)"].round().max() + 1)
    times = np.sort(times)
    recon_cols = [*[f"lon_{t}" for t in times], *[f"lat_{t}" for t in times]]

    df_recon = pd.DataFrame(columns=recon_cols, index=data.index, dtype=np.float_)
    data = data.join(df_recon)

    if not isinstance(rotation_model, pygplates.RotationModel):
        rotation_model = pygplates.RotationModel(rotation_model)
    topological_features = pygplates.FeaturesFunctionArgument(
        topological_features
    ).get_features()

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
            try:
                points = pygplates.MultiPointOnSphere(np.column_stack((lats, lons)))
            except pygplates.InsufficientPointsForMultiPointConstructionError as err:
                raise RuntimeError(
                    f"Reconstruction failed at time {t}"
                ) from err
            reconstructed_points = reconstruct_points(
                rotation_model,
                topological_features,
                reconstruction_begin_time=t,
                reconstruction_end_time=t - 1,
                reconstruction_time_interval=1.0,
                points=points,
                detect_collisions=None,
            )
            new_lats, new_lons = zip(*[i.to_lat_lon() for i in reconstructed_points])
            for i, new_lon, new_lat in zip(subset.index, new_lons, new_lats):
                data.at[i, new_lon_col] = new_lon
                data.at[i, new_lat_col] = new_lat
                
    return data


def load_data(
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

# --------------------------------------------------

def generate_grid_points(
    times,
    resolution,
    polygons_dir,
    plate_reconstruction=None,
    topological_features=None,
    rotation_model=None,
    output_filename=None,
    n_jobs=1,
    verbose=False,
):

    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    elif n_jobs < 0:
        n_jobs = cpu_count() + n_jobs + 1

    # Earlier times take longer to reconstruct, so ensure they are
    # evenly split between processes
    times = np.array(times)
    times_split = [
        times[i::n_jobs]
        for i in range(n_jobs)
    ]

    if plate_reconstruction is None:
        plate_reconstruction = PlateReconstruction(
            topology_features=topological_features,
            rotation_model=rotation_model,
        )

    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        out = parallel(
            delayed(_grid_points_subset)(
                times=t,
                resolution=resolution,
                polygons_dir=polygons_dir,
                plate_reconstruction=plate_reconstruction,
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


def _grid_points_subset(
    times,
    resolution,
    polygons_dir,
    plate_reconstruction,
    verbose=False,
):
    
    out = [
        _grid_points_time(
            time=t,
            resolution=resolution,
            polygons_dir=polygons_dir,
            plate_reconstruction=plate_reconstruction,
            verbose=verbose,
        )
        for t in times
    ]
    out = pd.concat(out, ignore_index=True)
    
    return out


def _grid_points_time(
    time,
    resolution,
    polygons_dir,
    plate_reconstruction,
    verbose=False,
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
        present_day_coords = reconstruct_by_topologies(
            data=pd.DataFrame(
                {
                    "lon": plons,
                    "lat": plats,
                    "age (Ma)": time,
                }
            ),
            plate_reconstruction=plate_reconstruction,
            times=np.arange(np.around(time) + 1),
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

# --------------------------------------------------

def prepare_deposit_data(
    deposit_data,
    plate_reconstruction,
    buffer_zones_dir,
    output_filename,
    time_steps,
    min_time=-np.inf,
    max_time=np.inf,
    n_jobs=1,
    verbose=False,
):
    
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
        plate_reconstruction=plate_reconstruction,
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


def prepare_unlabelled_data(
    unlabelled_data,
    plate_reconstruction,
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
        plate_reconstruction=plate_reconstruction,
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


def _get_overriding_plate_ids(
    data,
    plate_reconstruction,
    n_jobs=1,
    verbose=False,
):
    
    gdf = data.copy()
    geoms = []
    for _, row in data.iterrows():
        p = Point(row["lon"], row["lat"])
        geoms.append(p)
    gdf["geometry"] = geoms
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    times = data["age (Ma)"].unique()
    times_split = np.array_split(times, n_jobs)
    with Parallel(n_jobs, verbose=int(verbose)) as parallel:
        results = parallel(
            delayed(_overriding_plate_multiple_timesteps)(
                gdf=gdf[gdf["age (Ma)"].isin(t)],
                plate_reconstruction=plate_reconstruction,
            )
            for t in times_split
        )
    out = []
    for i in results:
        out.extend(i)
    out = pd.concat(out, ignore_index=True)
    columns_to_keep = set(
        list(data.columns.values) + ["overriding_plate_id"]
    )
    columns_to_drop = set(out.columns.values) - columns_to_keep
    out = out.drop(columns=list(columns_to_drop), errors="ignore")
    
    return out


def _overriding_plate_multiple_timesteps(gdf, plate_reconstruction):
    
    topological_features = plate_reconstruction.topology_features
    rotation_model = plate_reconstruction.rotation_model

    times = gdf["age (Ma)"].unique()
    out = []
    for time in times:
        out.append(
            _overriding_plate_timestep(
                gdf=gdf,
                topological_features=topological_features,
                rotation_model=rotation_model,
                time=time,
            )
        )
        
    return out


def _overriding_plate_timestep(
    gdf,
    topological_features,
    rotation_model,
    time,
):
    
    gdf = gpd.GeoDataFrame(gdf)

    if not isinstance(topological_features, pygplates.FeatureCollection):
        topological_features = pygplates.FeatureCollection(
            pygplates.FeaturesFunctionArgument(
                topological_features
            ).get_features()
        )
    if not isinstance(rotation_model, pygplates.RotationModel):
        rotation_model = pygplates.RotationModel(rotation_model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        reconstruction = PlateReconstruction(
            rotation_model=rotation_model,
            topology_features=topological_features,
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

# --------------------------------------------------

def run_coregister_point_data(
    point_data: _PathOrDataFrame,
    subduction_data: _PathOrDataFrame,
    output_filename: Optional[_PathLike] = None,
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
            delayed(coregister_point_data)(
                time=time,
                points=point_data[point_data["age (Ma)"] == time],
                szs=subduction_data[
                    subduction_data["age (Ma)"] == int(np.around(time))
                ],
            )
            for time in times
        )

    out = pd.DataFrame(pd.concat(out, ignore_index=True))

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


def coregister_point_data(
    time: float,
    points: pd.DataFrame,
    szs: pd.DataFrame,
) -> pd.DataFrame:

    points = points.copy()
    szs = szs.copy().reset_index()

    points = points[points["age (Ma)"] == time]
    szs = szs[szs["age (Ma)"] == int(np.around(time))]

    columns_to_add = set(szs.columns.values) - set(points.columns.values)
    for column in columns_to_add:
        points[column] = np.nan

    lon_points = np.array(points["lon"]).reshape((-1, 1))
    lat_points = np.array(points["lat"]).reshape((-1, 1))
    coords_points = np.deg2rad(np.hstack((lat_points, lon_points)))

    lon_data = np.array(szs["lon"]).reshape((-1, 1))
    lat_data = np.array(szs["lat"]).reshape((-1, 1))
    coords_data = np.deg2rad(np.hstack((lat_data, lon_data)))

    neigh = NearestNeighbors(metric="haversine", n_jobs=1)
    neigh.fit(coords_data)

    distances, indices = neigh.kneighbors(
        coords_points, n_neighbors=1, return_distance=True
    )
    # distances = np.rad2deg(distances).flatten()
    distances = distances.flatten() * EARTH_RADIUS
    indices = indices.flatten()

    for column in columns_to_add:
        for i_points, i_szs in zip(points.index, indices):
            points.at[i_points, column] = szs.at[i_szs, column]
    points["distance_to_trench (km)"] = distances

    return points

# --------------------------------------------------

def run_coregister_crustal_thickness(
    point_data: _PathOrDataFrame,
    input_dir: _PathLike,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
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
            delayed(coregister_crustal_thickness)(
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


def coregister_crustal_thickness(
    time: float,
    input_dir: _PathLike,
    df: _PathOrDataFrame,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
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

# --------------------------------------------------

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

# --------------------------------------------------

def downsample(
        df,
        n_target,
        n_bins_age=50,
        n_bins_spatial=10,
        random_state=None,
    ):
    
    # Determine target sample size
    if len(df) < n_target:
        print("No downsampling required!")
        return df
    
    if not isinstance(random_state, np.random.Generator):
        random_state = np.random.default_rng(random_state)
    
    df = df.copy()
    
    # Bin by age (quantile bins)
    df['age_bin'] = pd.qcut(df['age (Ma)'], q=n_bins_age, duplicates='drop')
    
    # Within each age bin, spatially bin and sample
    sampled_indices = []
    
    for age_bin, age_group in df.groupby('age_bin'):
        age_group = age_group.copy()
    
        # Spatial binning within this age group
        age_group['lon_bin'] = pd.cut(age_group['lon'], bins=n_bins_spatial)
        age_group['lat_bin'] = pd.cut(age_group['lat'], bins=n_bins_spatial)
    
        # Group by spatial bins
        spatial_groups = age_group.groupby(['lon_bin', 'lat_bin'])
    
        # Evenly sample from spatial bins in this age bin
        n_total_bins = len(spatial_groups)
        if n_total_bins == 0:
            continue
        samples_per_bin = max(1, (n_target // n_bins_age) // n_total_bins)
    
        for _, spatial_group in spatial_groups:
            n_samples = min(samples_per_bin, len(spatial_group))
            sampled_indices.extend(random_state.choice(spatial_group.index, size=n_samples, replace=False))
    
    # Return downsampled result
    sampled_df = df.loc[sampled_indices]
    sampled_df = sampled_df.drop(columns=['age_bin', 'lon_bin', 'lat_bin'], errors='ignore')

    return sampled_df
    

def analyze_correlations(corr_matrix, threshold=0.8):
    
    if isinstance(corr_matrix, str):
        corr_matrix = pd.read_csv(corr_matrix, index_col=0)
    else:
        corr_matrix = pd.DataFrame(corr_matrix)
    
    # Dictionary to store correlations
    correlations = {}
    
    for column in corr_matrix.columns:
        positive_corr = []
        negative_corr = []
        feature = corr_matrix[column]
        
        for i in range(feature.shape[0]):
            if abs(feature.iloc[i]) >= threshold and feature.index[i] != column:
                if feature.iloc[i] > 0:
                    positive_corr.append((feature.index[i], feature.iloc[i]))
                else:
                    negative_corr.append((feature.index[i], feature.iloc[i]))
        
        if positive_corr or negative_corr:
            correlations[column] = {
                'positive': sorted(positive_corr, key=lambda x: x[1], reverse=True),
                'negative': sorted(negative_corr, key=lambda x: x[1])
            }
    
    return correlations


def generate_report(correlations, threshold):
    
    print(f"Correlation Analysis Report (Threshold: {threshold})")
    print("=" * 50)
    
    for feature, corr in correlations.items():
        print(f"\nFeature: {feature}")
        print("-" * 30)
        
        if corr['positive']:
            print("Positive Correlations:")
            for c, value in corr['positive']:
                print(f"  {c}: {value:.3f}")
        
        if corr['negative']:
            print("Negative Correlations:")
            for c, value in corr['negative']:
                print(f"  {c}: {value:.3f}")
    
    print("\nTotal features with strong correlations:", len(correlations))

# --------------------------------------------------

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


def create_grids(
    data,
    output_dir,
    resolution=None,
    extent=None,
    threads=1,
    verbose=False,
    column="probability",
    filename_format="probability_grid_{}Ma.nc",
):

    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        data = pd.DataFrame(data)

    times = data["age (Ma)"].unique()

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
    verbose=False,
    filename_format="probability_grid_{}Ma.nc",
):
    
    time = int(np.around(time))
    output_filename = os.path.join(
        output_dir, filename_format.format(time)
    )

    if extent is None:
        xmin = np.nanmin(data_lons)
        xmax = np.nanmax(data_lons)
        ymin = np.nanmin(data_lats)
        ymax = np.nanmax(data_lats)
    elif extent == "global":
        xmin, xmax, ymin, ymax = -180, 180, -90, 90
    else:
        xmin, xmax, ymin, ymax = extent

    if resolution is None:
        resx = np.nanmin(np.gradient(np.sort(np.unique(data_lons))))
        resy = np.nanmin(np.gradient(np.sort(np.unique(data_lats))))
    else:
        resx = resolution
        resy = resolution

    grid_lons = np.arange(xmin, xmax + resx, resx)
    grid_lats = np.arange(ymin, ymax + resy, resy)
    grid_mlons, grid_mlats = np.meshgrid(grid_lons, grid_lats)

    arr = np.full((grid_lats.size, grid_lons.size), np.nan, dtype=float)
    for data_lon, data_lat, data_value in zip(
        data_lons, data_lats, data_values
    ):
        mask = np.logical_and(grid_mlons == data_lon, grid_mlats == data_lat)
        arr[mask] = data_value

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
    dset.rio.write_crs(4326, inplace=True)
    if verbose:
        print(
            "\t- Writing output file: " + os.path.basename(output_filename),
            file=stderr,
        )
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


def get_plot_topologies(
    model_name: Optional[str] = None,
    model_dir: str = "plate_model",
    anchor_plate_id: int = 0,
    time: Optional[int] = None,
    plate_reconstruction: Optional[PlateReconstruction] = None,
    filter_topologies: bool = False,
):
    
    if plate_reconstruction is None:
        plate_reconstruction = get_plate_reconstruction(
            model_name=model_name,
            model_dir=model_dir,
            anchor_plate_id=anchor_plate_id,
            filter_topologies=filter_topologies,
        )
    
    if model_name is None:
        file_exts = ["*.gpml", "*.gpmlz"]
        coastlines = []
        continents = []
        COBs = []
        for g in file_exts:
            filenames = glob.glob(os.path.join(model_dir, "**", g), recursive=True)
            for filename in filenames:
                basename = os.path.basename(filename)
                if "coast" in basename.lower():
                    coastlines.append(filename)
                elif "continent" in basename.lower():
                    continents.append(filename)
                elif "cob" in basename.lower():
                    COBs.append(filename)
    else:
        model = _fetch(model_name, os.path.join(model_dir, ".downloaded"))
        # pmm keys: 'Coastlines', 'StaticPolygons', 'ContinentalPolygons', 'Topologies', 'COBs', 'Cratons'
        coastlines = model.get_layer("Coastlines")
        continents = model.get_layer("ContinentalPolygons")
        COBs = model.get_layer("COBs")

    return PlotTopologies(
        plate_reconstruction=plate_reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=COBs,
        anchor_plate_id=anchor_plate_id,
        time=time,
    )


def calculate_entropy(probabilities):
 
    # Ensure probabilities are within [0,1] range
    probabilities = np.clip(probabilities, 1e-15, 1-1e-15)
    
    # For binary classification, get both class probabilities
    p_class1 = probabilities
    p_class0 = 1 - p_class1
    
    # Entropy calculation: -sum(p_i * log2(p_i))
    entropy = -p_class0 * np.log2(p_class0) - p_class1 * np.log2(p_class1)
    
    return entropy


def calculate_tree_vote_variance(rf_model, X, n_jobs=-1, verbose=1):

    n_trees = len(rf_model.estimators_)
    
    # Function to get predictions from a single tree
    def get_tree_predictions(tree_idx):
        
        tree = rf_model.estimators_[tree_idx]
        
        return tree.predict(X)
    
    # Use tqdm to track progress if verbose
    if verbose:
        tree_indices = tqdm(range(n_trees), desc="Processing trees")
    else:
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

