import glob
import os
from sys import stderr
from tempfile import NamedTemporaryFile
from typing import (
    Iterable,
    Optional,
    Sequence,
    Union,
)

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from plate_model_manager import PlateModelManager
import pygplates
from gplately import (
    PlateReconstruction,
    EARTH_RADIUS,
)

_PathLike = Union[os.PathLike, str]
INCREMENT = 1

PMM = PlateModelManager()


def run_calculate_convergence(
    nprocs: int,
    min_time: float,
    max_time: float,
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

    times = np.arange(min_time, max_time + INCREMENT, INCREMENT)
    
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
        tf = NamedTemporaryFile(suffix=".gpml")
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
