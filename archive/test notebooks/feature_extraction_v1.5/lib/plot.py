import os
import subprocess

import cartopy.crs as ccrs
import cmcrameri.cm as ccm
import geopandas as gpd
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from moviepy.config import FFMPEG_BINARY
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np
from tqdm import tqdm

from gplately import (
    PlateReconstruction,
    PlotTopologies,
)


def _generate_feat_map(gplot, subduction_data, projection, time, feature, output_dir):
    
    feat_min = subduction_data[feature].quantile(0.01)
    feat_max = subduction_data[feature].quantile(0.99)
    subduction_data_t = subduction_data[subduction_data["age (Ma)"] == time]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=-45, central_latitude=-45), facecolor="azure")

    gplot.plot_continents(ax, edgecolor="none", facecolor="tan", alpha=0.5, zorder=1)
    gplot.plot_coastlines(ax, edgecolor="none", facecolor="tan", alpha=0.7, zorder=2)
    gplot.plot_plate_motion_vectors(ax, spacingX=10, spacingY=10, normalise=True, alpha=0.1, zorder=3)

    feat = ax.scatter(subduction_data_t["lon"], subduction_data_t["lat"], 50, marker=".",
                      c=subduction_data_t[feature], cmap=ccm.hawaii_r, vmin=feat_min, vmax=feat_max,
                      transform=ccrs.PlateCarree(), zorder=4)
    
    # gplot.plot_all_topologies(ax, color="orangered", zorder=5)
    gplot.plot_ridges_and_transforms(ax, color="orangered", zorder=5)
    gplot.plot_trenches(ax, color="dimgray", zorder=6)
        
    gplot.plot_subduction_teeth(ax, spacing=0.05, color='black', alpha=0.3, zorder=7)

    ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color="gray", alpha=0.3, linestyle="--", zorder=8)
    
    num_ticks = 6
    ticks = np.linspace(feat_min, feat_max, num_ticks)
    
    cbar_ax = fig.add_axes([0.05, 0.25, 0.03, 0.5])  # [left, bottom, width, height]
    cbar_feat = fig.colorbar(feat, cax=cbar_ax, orientation="vertical", extend="both", ticks=ticks)
    cbar_feat.ax.yaxis.set_ticks_position('left')
    cbar_feat.ax.yaxis.set_label_position('left')
    cbar_feat.set_label(format_feature_name(feature), fontsize=14)
    cbar_feat.ax.tick_params(labelsize=12)
        
    custom_handles = [
        Patch(facecolor="tan", edgecolor="none", label="Continental Crust"),
        Line2D([0], [0], color="orangered", label="Mid-Ocean Ridges"),
        Line2D([0], [0], color="dimgray", label="Trench Lines"),
    ]
    ax.legend(handles=custom_handles, fontsize=14, loc="lower left", bbox_to_anchor=(-0.25, -0.05))
    ax.set_title(f"{time} Ma", fontsize=25, y=1.04)

    filename = os.path.join(output_dir, f"feat_map_{time:.0f}Ma.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
def _generate_feat_maps_parallel(
    rotation_model,
    topology_features,
    static_polygons,
    coastlines,
    continents,
    COBs,
    subduction_data,
    projection,
    time,
    feature,
    output_dir,
    anchor_plate_id,
    ):
    
    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
    )
    
    gplot = PlotTopologies(
        plate_reconstruction=plate_reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=COBs,
        anchor_plate_id=anchor_plate_id,
        time=time,
    )
    
    return _generate_feat_map(
        gplot,
        subduction_data,
        projection,
        time,
        feature,
        output_dir,
    )


def generate_feat_maps(
        rotation_model,
        topology_features,
        static_polygons,
        coastlines,
        continents,
        COBs,
        subduction_data,
        projection,
        time_steps,
        feature,
        output_dir,
        anchor_plate_id=0,
        n_jobs=-2
        ):
    
    tasks = (delayed(_generate_feat_maps_parallel)(
        rotation_model,
        topology_features,
        static_polygons,
        coastlines,
        continents,
        COBs,
        subduction_data,
        projection,
        time,
        feature,
        output_dir,
        anchor_plate_id,
        ) for time in tqdm(time_steps, desc="Dispatching tasks"))
    
    Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    

def _generate_buffer_zone_map(gplot, projection, time, buffer_zones_dir, output_dir):
    
    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=projection, facecolor="azure")

    gplot.plot_continents(ax, edgecolor="none", facecolor="tan", alpha=0.5, zorder=1)
    gplot.plot_coastlines(ax, edgecolor="none", facecolor="tan", alpha=0.7, zorder=2)
    gplot.plot_plate_motion_vectors(ax, spacingX=10, spacingY=10, normalise=True, alpha=0.1, zorder=3)

    buffer_zones_t_filename = f'buffer_zones_{time}Ma.geojson'
    buffer_zones_t_filename = os.path.join(buffer_zones_dir, buffer_zones_t_filename)
    buffer_zones_t = gpd.read_file(buffer_zones_t_filename)

    buffer_zones_t.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        facecolor='palegreen',
        edgecolor='none',
        alpha=0.7,
        zorder=4,
    )

    # gplot.plot_all_topologies(ax, color="orangered", zorder=5)
    gplot.plot_ridges_and_transforms(ax, color="orangered", zorder=5)
    gplot.plot_trenches(ax, color="black", zorder=6)
    gplot.plot_subduction_teeth(ax, spacing=0.05, color="black", zorder=7)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False,
                      linewidth=1, color="gray", alpha=0.3, linestyle="--", zorder=8)
    
    ax.text(0.49,-0.03, "60°E", transform=ax.transAxes, fontsize=16)
    ax.text(0.46,-0.03, "0°", transform=ax.transAxes, fontsize=16)
    ax.text(0.40,-0.025, "60°W", transform=ax.transAxes, fontsize=16)

    gl.top_labels = False
    gl.bottom_labels = False
    gl.xlabel_style = {"size": 16}
    gl.ylabel_style = {"size": 16}

    custom_handles = [
        Patch(facecolor='tan', edgecolor='none', label='Continental Crust'),
        Patch(facecolor='palegreen', edgecolor='none', alpha=0.7, label='Target Areas in\nBack-Arc Basins'),
        Line2D([0], [0], color="orangered", label="Mid-Ocean Ridges"),
        Line2D([0], [0], color="black", label="Trench Lines")
    ]

    ax.legend(handles=custom_handles, fontsize=16, loc="lower left", bbox_to_anchor=(0, -0.2))
    ax.set_title(f"{time} Ma", fontsize=25, y=1.04)

    filename = os.path.join(output_dir, f"buffer_zone_map_{time:.0f}Ma.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    
def _generate_buffer_zone_maps_parallel(
    rotation_model,
    topology_features,
    static_polygons,
    coastlines,
    continents,
    COBs,
    projection,
    time,
    buffer_zones_dir,
    output_dir,
    anchor_plate_id,
    ):
    
    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
    )
    
    gplot = PlotTopologies(
        plate_reconstruction=plate_reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=COBs,
        anchor_plate_id=anchor_plate_id,
        time=time,
    )
    
    return _generate_buffer_zone_map(
        gplot,
        projection,
        time,
        buffer_zones_dir,
        output_dir,
    )


def generate_buffer_zone_maps(
        rotation_model,
        topology_features,
        static_polygons,
        coastlines,
        continents,
        COBs,
        projection,
        time_steps,
        buffer_zones_dir,
        output_dir,
        anchor_plate_id=0,
        n_jobs=-2
        ):
    
    tasks = (delayed(_generate_buffer_zone_maps_parallel)(
        rotation_model,
        topology_features,
        static_polygons,
        coastlines,
        continents,
        COBs,
        projection,
        time,
        buffer_zones_dir,
        output_dir,
        anchor_plate_id,
        ) for time in tqdm(time_steps, desc="Dispatching tasks"))
    
    Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    

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
