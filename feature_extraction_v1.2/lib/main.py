import glob
import os
import subprocess
from sys import stderr
from tempfile import NamedTemporaryFile
from typing import (
    Iterable,
    Optional,
    Sequence,
    Union,
)

import cartopy.crs as ccrs
import cmcrameri.cm as ccm
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from moviepy.config import FFMPEG_BINARY
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np
import pandas as pd
from tqdm import tqdm

from plate_model_manager import PlateModelManager
import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    EARTH_RADIUS,
)

_PathLike = Union[os.PathLike, str]
INCREMENT = 1

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
        # pmm keys: "Coastlines", "StaticPolygons", "ContinentalPolygons", "Topologies", "COBs", "Cratons"
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


def generate_feat_map(gplot, subduction_data, projection, time, feature, output_dir):
    
    gplot.time = time

    feat_min = subduction_data[feature].min()
    feat_max = subduction_data[feature].max()
    subduction_data_t = subduction_data[subduction_data["age (Ma)"] == time]

    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=projection, facecolor="azure")

    gplot.plot_continents(ax, edgecolor="none", facecolor="tan", alpha=0.5, zorder=1)
    gplot.plot_coastlines(ax, edgecolor="none", facecolor="tan", alpha=0.7, zorder=2)
    gplot.plot_plate_motion_vectors(ax, spacingX=10, spacingY=10, normalise=True, alpha=0.1, zorder=3)

    feat = ax.scatter(subduction_data_t["lon"], subduction_data_t["lat"], 50, marker=".",
                      c=subduction_data_t[feature], cmap=ccm.hawaii_r, vmin=feat_min, vmax=feat_max,
                      transform=ccrs.PlateCarree(), zorder=4)

    gplot.plot_all_topologies(ax, color="dimgray", linewidth=1.5, zorder=5)
    gplot.plot_trenches(ax, color="k", alpha=0.3, zorder=6)
    gplot.plot_subduction_teeth(ax, spacing=0.05, color="k", alpha=0.3, zorder=7)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False,
                      linewidth=1, color="gray", alpha=0.3, linestyle="--", zorder=8)
    
    ax.text(0.49,-0.03, "60°E", transform=ax.transAxes, fontsize=16)
    ax.text(0.46,-0.03, "0°", transform=ax.transAxes, fontsize=16)
    ax.text(0.40,-0.025, "60°W", transform=ax.transAxes, fontsize=16)

    gl.top_labels = False
    gl.bottom_labels = False
    gl.xlabel_style = {"size": 16}
    gl.ylabel_style = {"size": 16}

    num_ticks = 6
    ticks = np.linspace(feat_min, feat_max, num_ticks)
        
    cbar_feat = fig.colorbar(feat, shrink=0.4, pad=0.06, orientation="horizontal", extend="both", ticks=ticks)
    cbar_feat.set_label(feature, fontsize=16, labelpad=10)
    cbar_feat.ax.tick_params(labelsize=16)
    cbar_feat.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    custom_handles = [
        Patch(facecolor="tan", edgecolor="none", label="Continental Crust"),
        Line2D([0], [0], color="dimgray", lw=2, label="Mid-Ocean Ridge")
    ]
    ax.legend(handles=custom_handles, fontsize=16, loc="lower left", bbox_to_anchor=(0, -0.2))
    ax.set_title(f"{time} Ma", fontsize=25, y=1.04)

    filename = os.path.join(output_dir, f"feat_map_{time:.0f}Ma.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    

def generate_feat_maps(gplot, subduction_data, projection, time_steps, feature, output_dir, num_jobs=-1):
    
    tasks = (delayed(generate_feat_map)(gplot, subduction_data, projection, time, feature, output_dir) for time in tqdm(time_steps, desc="Dispatching tasks"))
    Parallel(n_jobs=num_jobs, backend="loky")(tasks)


def create_animation(
    image_filenames,
    output_filename,
    fps=5,
    codec="auto",
    bitrate="5000k",
    output_fps=30,
    ffmpeg_params=None,
    **kwargs
):
    
    if codec == "hevc":
        if hwaccel_available():
            codec = "hevc_videotoolbox"
        else:
            codec = "hevc"
    elif codec == "auto":
        codec = "libx264"

    if ffmpeg_params is None:
        ffmpeg_params = [
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ]

    logger = kwargs.pop("logger", None)
    audio = kwargs.pop("audio", False)

    with ImageSequenceClip(image_filenames, fps=fps) as clip:
        clip.write_videofile(
            output_filename,
            fps=output_fps,
            codec=codec,
            bitrate=bitrate,
            audio=audio,
            logger=logger,
            ffmpeg_params=ffmpeg_params,
            **kwargs,
        )


def hwaccel_available(codec="hevc_videotoolbox"):
    
    return codec_available(codec)


def codec_available(codec):
    
    result = _test_codec(codec)
    
    return result.returncode == 0

def _test_codec(codec):
    
    cmd = [
        FFMPEG_BINARY,
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=color=black:size=1080x1080",
        "-vframes", "1",
        "-pix_fmt", "yuv420p10le",
        "-an",
        "-c:v", codec,
        "-f", "null",
        "-",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
    )
    
    return result

