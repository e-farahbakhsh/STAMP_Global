'''
Plot

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 05/07/2025
'''


import subprocess
from typing import (
    Any,
    Mapping,
    Optional,
    Union,
)

import cartopy.crs as ccrs
import cmcrameri.cm as ccm
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from moviepy.config import FFMPEG_BINARY
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np
import pandas as pd

import pygplates
from gplately import (
    PlateReconstruction,
    PlotTopologies,
    Raster,
)

from lib.reconstruction_coregistration import reconstruct_by_topologies


FIGURE_SIZE_ORTHOGRAPHIC = (10, 10)
FIGURE_SIZE_MOLLWEIDE = (16, 12)
FONT_SIZE = 16
TICK_SIZE = 16
TITLE_SIZE = 25
SUPTITLE_SIZE = 30
BACKGROUND_COLOUR = "darkgray"
TESSELLATE_DEGREES = 0.1


BACKGROUND_KWARGS = {
    "cmap": ccm.lapaz_r,
    "vmin": 0,
    "vmax": 230,
    "alpha": 0.7,
    "zorder": 1,
}
CONTINENTS_KWARGS = {
    "edgecolor": "none",
    "facecolor": "darkgray",
    "alpha": 0.5,
    "zorder": BACKGROUND_KWARGS["zorder"] + 1,
    "tessellate_degrees": TESSELLATE_DEGREES,
}
COASTLINES_KWARGS = {
    "edgecolor": "none",
    "facecolor": "darkgray",
    "alpha": 1,
    "zorder": BACKGROUND_KWARGS["zorder"] + 1,
    "tessellate_degrees": TESSELLATE_DEGREES,
}
VECTORS_KWARGS = {
    "spacingX": 10,
    "spacingY": 10,
    "normalise": True,
    "alpha": 0.1,
    "zorder": CONTINENTS_KWARGS["zorder"] + 1,
}
RIDGES_KWARGS = {
    "color": "dimgray",
    "linewidth": 1.5,
    "alpha": 1,
    "zorder": VECTORS_KWARGS["zorder"] + 1,
    "tessellate_degrees": TESSELLATE_DEGREES,
}
TOPOLOGIES_KWARGS = {
    "color": "dimgray",
    "linewidth": 1.5,
    "alpha": 1,
    "zorder": VECTORS_KWARGS["zorder"] + 1,
    "tessellate_degrees": TESSELLATE_DEGREES,
}
PROBS_KWARGS = {
    "cmap": ccm.hawaii_r,
    "vmin": 0,
    "vmax": 100,
    "alpha": 0.9,
    "zorder": RIDGES_KWARGS["zorder"] + 1,
}
ENTROPY_KWARGS = {
    "cmap": ccm.hawaii_r,
    "vmin": 0,
    "vmax": 1,
    "alpha": 0.9,
    "zorder": RIDGES_KWARGS["zorder"] + 1,
}
VOTE_VAR_KWARGS = {
    "cmap": ccm.hawaii_r,
    "vmin": 0,
    "vmax": 0.25,
    "alpha": 0.9,
    "zorder": RIDGES_KWARGS["zorder"] + 1,
}

hawaii = ccm.hawaii
single_color = hawaii(0.0)
n_colors_total = 256
n_colors_single = 128
n_colors_hawaii = 128
lower_colors = np.tile(single_color, (n_colors_single, 1))
upper_colors = hawaii(np.linspace(0, 1, n_colors_hawaii))
combined_colors = np.vstack([lower_colors, upper_colors])
custom_preservation_cmap = ListedColormap(combined_colors, name='custom_preservation')

PRES_KWARGS = {
    "cmap": custom_preservation_cmap,
    "vmin": 0,
    "vmax": 100,
    "alpha": 0.9,
    "zorder": RIDGES_KWARGS["zorder"] + 1,
}

TRENCHES_KWARGS = {
    "color": "black",
    "alpha": 0.3,
    "zorder": PROBS_KWARGS["zorder"] + 1,
    "tessellate_degrees": TESSELLATE_DEGREES,
}
TEETH_KWARGS = {
    # "size": 8,
    # "aspect": 0.7,
    "spacing": 0.05,
    # "markerfacecolor": "black",
    # "markeredgecolor": "black",
    "facecolor": "black",
    "edgecolor": "black",
    "alpha": 0.3,
    "zorder": TRENCHES_KWARGS["zorder"] + 1,
}
SCATTER_KWARGS = {
    "marker": "o",
    "facecolor": "yellow",
    "edgecolor": "black",
    "transform": ccrs.PlateCarree(),
    "zorder": TEETH_KWARGS["zorder"] + 1,
}
GRIDS_KWARGS = {
    "crs": ccrs.PlateCarree(),
    "draw_labels": True,
    "x_inline": False,
    "linewidth": 1,
    "color": "gray",
    "alpha": 0.3,
    "linestyle": "--",
    "zorder": SCATTER_KWARGS["zorder"] + 1,
    }
SAVEFIG_KWARGS = {
    "dpi": 200,
    "bbox_inches": "tight",
}


def plot_parallel(
    rotation_model,
    topology_features,
    static_polygons,
    coastlines,
    continents,
    COBs,
    grid,
    grid_type,
    background_grid,
    projection,
    time,
    positives,
    output_filename,
    central_meridian,
    background_kwargs: Optional[Mapping[str, Any]] = None,
    continents_kwargs: Optional[Mapping[str, Any]] = None,
    coastlines_kwargs: Optional[Mapping[str, Any]] = None,
    ridges_kwargs: Optional[Mapping[str, Any]] = None,
    topologies_kwargs: Optional[Mapping[str, Any]] = None,
    vectors_kwargs: Optional[Mapping[str, Any]] = None,
    probs_kwargs: Optional[Mapping[str, Any]] = None,
    entropy_kwargs: Optional[Mapping[str, Any]] = None,
    vote_var_kwargs: Optional[Mapping[str, Any]] = None,
    pres_kwargs: Optional[Mapping[str, Any]] = None,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    trenches_kwargs: Optional[Mapping[str, Any]] = None,
    teeth_kwargs: Optional[Mapping[str, Any]] = None,
    grids_kwargs: Optional[Mapping[str, Any]] = None,
):
    
    plate_reconstruction = PlateReconstruction(rotation_model, topology_features, static_polygons)
    gplot = PlotTopologies(
        plate_reconstruction=plate_reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=COBs,
        time=time,
    )

    # Call the original plotting function
    return plot(
        gplot=gplot,
        grid=grid,
        grid_type=grid_type,
        background_grid=background_grid,
        projection=projection,
        time=time,
        positives=positives,
        output_filename=output_filename,
        central_meridian=central_meridian,
        background_kwargs=background_kwargs,
        continents_kwargs=continents_kwargs,
        coastlines_kwargs=coastlines_kwargs,
        ridges_kwargs=ridges_kwargs,
        topologies_kwargs=topologies_kwargs,
        vectors_kwargs=vectors_kwargs,
        probs_kwargs=probs_kwargs,
        entropy_kwargs=entropy_kwargs,
        vote_var_kwargs=vote_var_kwargs,
        pres_kwargs=pres_kwargs,
        scatter_kwargs=scatter_kwargs,
        trenches_kwargs=trenches_kwargs,
        teeth_kwargs=teeth_kwargs,
        grids_kwargs=grids_kwargs,
        )


def plot(
    gplot,
    grid,
    grid_type,
    background_grid,
    projection=None,
    time=None,
    positives=None,
    output_filename=None,
    central_meridian=None,
    background_kwargs: Optional[Mapping[str, Any]] = None,
    continents_kwargs: Optional[Mapping[str, Any]] = None,
    coastlines_kwargs: Optional[Mapping[str, Any]] = None,
    ridges_kwargs: Optional[Mapping[str, Any]] = None,
    topologies_kwargs: Optional[Mapping[str, Any]] = None,
    vectors_kwargs: Optional[Mapping[str, Any]] = None,
    probs_kwargs: Optional[Mapping[str, Any]] = None,
    entropy_kwargs: Optional[Mapping[str, Any]] = None,
    vote_var_kwargs: Optional[Mapping[str, Any]] = None,
    pres_kwargs: Optional[Mapping[str, Any]] = None,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    trenches_kwargs: Optional[Mapping[str, Any]] = None,
    teeth_kwargs: Optional[Mapping[str, Any]] = None,
    grids_kwargs: Optional[Mapping[str, Any]] = None,
):

    coastlines_kwargs = {} if coastlines_kwargs is None else coastlines_kwargs
    coastlines_kwargs = _copy_update_dict(COASTLINES_KWARGS, coastlines_kwargs)
    
    ridges_kwargs = {} if ridges_kwargs is None else ridges_kwargs
    ridges_kwargs = _copy_update_dict(RIDGES_KWARGS, ridges_kwargs)

    # topologies_kwargs = {} if topologies_kwargs is None else topologies_kwargs
    # topologies_kwargs = _copy_update_dict(TOPOLOGIES_KWARGS, topologies_kwargs)
    
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs = _copy_update_dict(SCATTER_KWARGS, scatter_kwargs)

    if not isinstance(gplot, PlotTopologies):
        gplot = _get_gplot(**gplot)

    if time is not None and gplot.time != time:
        gplot.time = time

    if positives is not None:
        if isinstance(positives, str):
            positives = pd.read_csv(positives)
        else:
            positives = pd.DataFrame(positives)

    if projection is None or str(projection).lower() == "mollweide":
        projection = ccrs.Mollweide(
            central_meridian if central_meridian is not None else 0
        )
    elif str(projection).lower() == "orthographic":
        projection = ccrs.Orthographic(
            central_meridian if central_meridian is not None else 0,
            10,
        )
    if not isinstance(projection, ccrs.CRS):
        raise TypeError(f"Invalid projection {projection}")

    if isinstance(projection, ccrs.Mollweide):
        figsize = FIGURE_SIZE_MOLLWEIDE
    elif isinstance(projection, ccrs.Orthographic):
        figsize = FIGURE_SIZE_ORTHOGRAPHIC
    else:  # determine figure size automatically
        figsize = None

    raster = Raster(grid)
    
    if grid_type in ("probability", "preservation", "probability_pres"):
        raster.data *= 100.0
        
    raster_bg = Raster(background_grid)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(
        [0.1, 0.1, 0.8, 0.8],
        projection=projection,
        facecolor=(plt.cm.colors.to_rgba(BACKGROUND_COLOUR, alpha=0.5)),
    )

    bg, grid_ = _prepare_axes(
        ax=ax,
        gplot=gplot,
        raster=raster,
        grid_type=grid_type,
        raster_bg=raster_bg,
        time=time,
        positives=positives,
        central_meridian=central_meridian,
        background_kwargs=background_kwargs,
        continents_kwargs=continents_kwargs,
        coastlines_kwargs=coastlines_kwargs,
        ridges_kwargs=ridges_kwargs,
        topologies_kwargs=topologies_kwargs,
        vectors_kwargs=vectors_kwargs,
        probs_kwargs=probs_kwargs,
        entropy_kwargs=entropy_kwargs,
        vote_var_kwargs=vote_var_kwargs,
        pres_kwargs=pres_kwargs,
        scatter_kwargs=scatter_kwargs,
        trenches_kwargs=trenches_kwargs,
        teeth_kwargs=teeth_kwargs,
        grids_kwargs=grids_kwargs,
    )
    
    cax_grid = fig.add_axes([0.1, 0.12, 0.35, 0.02])
    cax_bg = fig.add_axes([0.55, 0.12, 0.35, 0.02])
            
    cbar_grid = fig.colorbar(grid_, cax=cax_grid, orientation="horizontal")
    if grid_type == "probability":
        cbar_grid.set_label("Mineralisation Probability (%)", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0, 20, 40, 60, 80, 100])
    elif grid_type == "entropy":
        cbar_grid.set_label("Entropy", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    elif grid_type == "vote variance":
        cbar_grid.set_label("Vote Variance", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    elif grid_type == "preservation":
        cbar_grid.set_label("Preservation Likelihood (%)", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0, 20, 40, 60, 80, 100])
    elif grid_type == "probability_pres":
        cbar_grid.set_label("Preserved Mineralisation Probability (%)", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0, 20, 40, 60, 80, 100])    

    cbar_grid.ax.tick_params(labelsize=TICK_SIZE)

    cbar_bg = fig.colorbar(bg, cax=cax_bg, orientation="horizontal", extend='max')
    cbar_bg.set_label("Seafloor Age (Ma)", fontsize=FONT_SIZE, labelpad=10)
    cbar_bg.set_ticks([0, 50, 100, 150, 200])
    cbar_bg.ax.tick_params(labelsize=TICK_SIZE)
    
    custom_handles = [
    Patch(facecolor=coastlines_kwargs["facecolor"], edgecolor=coastlines_kwargs["edgecolor"], alpha=coastlines_kwargs["alpha"], label="Continental Crust"),
    # Line2D([0], [0], color=topologies_kwargs["color"], lw=2, label="Mid-Ocean Ridges/\nTransform Faults"),
    Line2D([0], [0], color=ridges_kwargs["color"], lw=2, label="Mid-Ocean Ridges/\nTransform Faults"),
    Line2D([0], [0], marker=scatter_kwargs["marker"], markerfacecolor=scatter_kwargs["facecolor"], markeredgecolor=scatter_kwargs["edgecolor"], markersize=15, linestyle='None', label='Mineral Occurrences')
    ]
    ax.legend(handles=custom_handles, fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0, -0.15))
    
    if output_filename is not None:
        fig.savefig(output_filename, **SAVEFIG_KWARGS)
        plt.close(fig)
        
    return fig


def _copy_update_dict(old: dict, new: dict):
    
    tmp = dict(old).copy()
    tmp.update(dict(new))
    
    return tmp


def _get_gplot(
    rotation_model,
    topology_features=None,
    static_polygons=None,
    coastlines=None,
    continents=None,
    COBs=None,
    time=None,
    anchor_plate_id=0,
):
    
    reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=pygplates.FeatureCollection(
            [
                i for i in pygplates.FeaturesFunctionArgument(
                    topology_features
                ).get_features()
                if i.get_feature_type().to_qualified_string()
                != "gpml:TopologicalSlabBoundary"
            ]
    ),
        static_polygons=static_polygons,
    )
    gplot = PlotTopologies(
        plate_reconstruction=reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=COBs,
        time=time,
        anchor_plate_id=anchor_plate_id,
    )
    
    return gplot


def _prepare_axes(
    ax: Axes,
    gplot: PlotTopologies,
    raster: Raster,
    grid_type: str,
    raster_bg: Raster,
    time: float,
    positives: Optional[pd.DataFrame] = None,
    positives_window_size: float = 2.5,
    central_meridian: float = 0.0,
    background_kwargs: Optional[Mapping[str, Any]] = None,
    continents_kwargs: Optional[Mapping[str, Any]] = None,
    coastlines_kwargs: Optional[Mapping[str, Any]] = None,
    ridges_kwargs: Optional[Mapping[str, Any]] = None,
    topologies_kwargs: Optional[Mapping[str, Any]] = None,
    vectors_kwargs: Optional[Mapping[str, Any]] = None,
    probs_kwargs: Optional[Mapping[str, Any]] = None,
    entropy_kwargs: Optional[Mapping[str, Any]] = None,
    vote_var_kwargs: Optional[Mapping[str, Any]] = None,
    pres_kwargs: Optional[Mapping[str, Any]] = None,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    trenches_kwargs: Optional[Mapping[str, Any]] = None,
    teeth_kwargs: Optional[Mapping[str, Any]] = None,
    grids_kwargs: Optional[Mapping[str, Any]] = None,
    **kwargs,
):
    
    background_kwargs = {} if background_kwargs is None else background_kwargs
    background_kwargs = _copy_update_dict(BACKGROUND_KWARGS, background_kwargs)

    continents_kwargs = {} if continents_kwargs is None else continents_kwargs
    continents_kwargs = _copy_update_dict(CONTINENTS_KWARGS, continents_kwargs)

    coastlines_kwargs = {} if coastlines_kwargs is None else coastlines_kwargs
    coastlines_kwargs = _copy_update_dict(COASTLINES_KWARGS, coastlines_kwargs)

    ridges_kwargs = {} if ridges_kwargs is None else ridges_kwargs
    ridges_kwargs = _copy_update_dict(RIDGES_KWARGS, ridges_kwargs)

    topologies_kwargs = {} if topologies_kwargs is None else topologies_kwargs
    topologies_kwargs = _copy_update_dict(TOPOLOGIES_KWARGS, topologies_kwargs)

    vectors_kwargs = {} if vectors_kwargs is None else vectors_kwargs
    vectors_kwargs = _copy_update_dict(VECTORS_KWARGS, vectors_kwargs)
    
    if grid_type == "probability" or grid_type == "probability_pres":
        probs_kwargs = {} if probs_kwargs is None else probs_kwargs
        probs_kwargs = _copy_update_dict(PROBS_KWARGS, probs_kwargs)
    elif grid_type == "entropy":
        entropy_kwargs = {} if entropy_kwargs is None else entropy_kwargs
        entropy_kwargs = _copy_update_dict(ENTROPY_KWARGS, entropy_kwargs)
    elif grid_type == "vote variance":
        vote_var_kwargs = {} if vote_var_kwargs is None else vote_var_kwargs
        vote_var_kwargs = _copy_update_dict(VOTE_VAR_KWARGS, vote_var_kwargs)
    elif grid_type == "preservation":
        pres_kwargs = {} if pres_kwargs is None else pres_kwargs
        pres_kwargs = _copy_update_dict(PRES_KWARGS, pres_kwargs)

    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    scatter_kwargs = _copy_update_dict(SCATTER_KWARGS, scatter_kwargs)

    trenches_kwargs = {} if trenches_kwargs is None else trenches_kwargs
    trenches_kwargs = _copy_update_dict(TRENCHES_KWARGS, trenches_kwargs)

    teeth_kwargs = {} if teeth_kwargs is None else teeth_kwargs
    teeth_kwargs = _copy_update_dict(TEETH_KWARGS, teeth_kwargs)

    grids_kwargs = {} if grids_kwargs is None else grids_kwargs
    grids_kwargs = _copy_update_dict(GRIDS_KWARGS, grids_kwargs)
    
    bg = raster_bg.imshow(
        ax=ax,
        **background_kwargs,
        )
    
    # gplot.plot_continents(
    #     ax=ax,
    #     central_meridian=central_meridian,
    #     **continents_kwargs,
    #     )
    
    gplot.plot_coastlines(
        ax=ax,
        # central_meridian=central_meridian,
        **coastlines_kwargs,
        )
    
    gplot.plot_ridges_and_transforms(
        ax=ax,
        # central_meridian=central_meridian,
        **ridges_kwargs,
    )
    
    # gplot.plot_all_topologies(
    #     ax=ax,
    #     # central_meridian=central_meridian,
    #     **topologies_kwargs,
    #     )
    
    gplot.plot_plate_motion_vectors(
        ax=ax,
        **vectors_kwargs,
        )
    
    if grid_type == "probability" or grid_type == "probability_pres":
        probs = raster.imshow(
            ax=ax,
            **probs_kwargs,
            )
    elif grid_type == "entropy":
        entropy = raster.imshow(
            ax=ax,
            **entropy_kwargs,
            )
    elif grid_type == "vote variance":
        vote_var = raster.imshow(
            ax=ax,
            **vote_var_kwargs,
            )
    elif grid_type == "preservation":
        pres = raster.imshow(
            ax=ax,
            **pres_kwargs,
            )
            
    gplot.plot_trenches(
        ax=ax,
        # central_meridian=central_meridian,
        **trenches_kwargs,
        )
        
    gplot.plot_subduction_teeth(
        ax=ax,
        **teeth_kwargs,
        )

    if (
        positives is not None
        and isinstance(positives, pd.DataFrame)
        and positives.shape[0] > 0
    ):
        if (
            f"lon_{time:0.0f}" not in positives.columns
            or f"lat_{time:0.0f}" not in positives.columns
        ):
            positives = reconstruct_by_topologies(
                data=positives,
                plate_reconstruction=gplot.plate_reconstruction,
                times=np.arange(time, positives["age (Ma)"].round().max() + 1),
                verbose=False,
            )
        _add_deposits(
            ax=ax,
            deposits=positives,
            time=time,
            window_size=positives_window_size,
            **scatter_kwargs,
        )
    ax.set_global()
    
    gl = ax.gridlines(
        **grids_kwargs,
        )

    ax.text(0.49,-0.03, '60°E', transform=ax.transAxes, fontsize=TICK_SIZE)
    ax.text(0.46,-0.03, '0°', transform=ax.transAxes, fontsize=TICK_SIZE)
    ax.text(0.40,-0.025, '60°W', transform=ax.transAxes, fontsize=TICK_SIZE)
    
    gl.top_labels=False
    gl.bottom_labels=False
    
    gl.xlabel_style = {"size": TICK_SIZE}
    gl.ylabel_style = {"size": TICK_SIZE}

    if time is not None:
        ax.set_title(
            f"{time} Ma",
            fontsize=kwargs.get("fontsize", TITLE_SIZE),
            y=1.04,
        )

    if grid_type == "probability" or grid_type == "probability_pres":
        return bg, probs
    elif grid_type == "entropy":
        return bg, entropy
    elif grid_type == "vote variance":
        return bg, vote_var
    elif grid_type == "preservation":
        return bg, pres
        

def _add_deposits(
    ax: Axes,
    deposits: Union[str, pd.DataFrame],
    time: Optional[float] = None,
    window_size: float = 2.5,
    size_scale: float = 30.0,
    **kwargs,
):
    
    if not isinstance(deposits, pd.DataFrame):
        deposits = pd.read_csv(deposits)
    if "label" in deposits.columns:
        deposits = deposits[deposits["label"] == "positive"]

    if time is not None and "age (Ma)" in deposits.columns:
        if (
            f"lon_{time:0.0f}" in deposits.columns
            and f"lat_{time:0.0f}" in deposits.columns
        ):
            alpha = kwargs.pop("alpha", 1.0)
            zorder = kwargs.pop("zorder", 1)
            oldalpha = kwargs.pop("oldalpha", alpha * 0.25)
            oldzorder = kwargs.pop("oldzorder", zorder - 0.01)

            new_deposits = deposits[
                (deposits["age (Ma)"] >= time)
                & ((deposits["age (Ma)"] - time) <= window_size)
            ]
            old_deposits = deposits[
                deposits["age (Ma)"] > (time + window_size)
            ]
            
            if "weight" in new_deposits.columns:
                markersizes_new = new_deposits["weight"] * size_scale
            else:
                markersizes_new = size_scale

            if "weight" in old_deposits.columns:
                markersizes_old = old_deposits["weight"] * (size_scale * 0.5)
            else:
                markersizes_old = size_scale * 0.5
            
            out = []
            out.append(
                ax.scatter(
                    new_deposits[f"lon_{time:0.0f}"],
                    new_deposits[f"lat_{time:0.0f}"],
                    alpha=alpha,
                    s=markersizes_new,
                    zorder=zorder,
                    **kwargs,
                )
            )
            out.append(
                ax.scatter(
                    old_deposits[f"lon_{time:0.0f}"],
                    old_deposits[f"lat_{time:0.0f}"],
                    alpha=oldalpha,
                    s=markersizes_old,
                    zorder=oldzorder,
                    **kwargs,
                )
            )
            return out

        # Reconstructed coordinates do not exist, but time is not None
        deposits = deposits[
            (deposits["age (Ma)"] - time).abs() <= window_size
        ]
    
    markersizes = (
        deposits["weight"] * size_scale
        if "weight" in deposits.columns
        else size_scale
    )

    return ax.plot(
        deposits["lon"],
        deposits["lat"],
        s=markersizes,
        **kwargs,
    )


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
