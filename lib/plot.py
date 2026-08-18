'''
Plot

Author: Ehsan Farahbakhsh
Contact email: e.farahbakhsh@sydney.edu.au
Date last modified: 12/05/2026
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
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
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

from lib.reconstruction_coregistration import reconstruct_points_


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
    "linewidth": 1.2,
    "alpha": 1,
    "zorder": VECTORS_KWARGS["zorder"] + 1,
    "tessellate_degrees": TESSELLATE_DEGREES,
}
TOPOLOGIES_KWARGS = {
    "color": "dimgray",
    "linewidth": 1.2,
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
UNCERTAINTY_KWARGS = {
    "cmap": ccm.hawaii_r,
    "vmin": 0,
    "vmax": 1,
    "alpha": 0.9,
    "zorder": RIDGES_KWARGS["zorder"] + 1,
}
PRES_KWARGS = {
    "cmap": ccm.hawaii_r,
    "vmin": 0,
    "vmax": 100,
    "alpha": 0.9,
    "zorder": RIDGES_KWARGS["zorder"] + 1,
}
TRENCHES_KWARGS = {
    "color": "black",
    "alpha": 0.5,
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
    "alpha": 0.5,
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
    anchor_plate_id,
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
    uncertainty_kwargs: Optional[Mapping[str, Any]] = None,
    probs_unc_kwargs: Optional[Mapping[str, Any]] = None,
    pres_kwargs: Optional[Mapping[str, Any]] = None,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    trenches_kwargs: Optional[Mapping[str, Any]] = None,
    teeth_kwargs: Optional[Mapping[str, Any]] = None,
    grids_kwargs: Optional[Mapping[str, Any]] = None,
):
    
    """
    Generate a global tectonic reconstruction plot with parallel geological layers.

    This function wraps the lower-level `plot` routine by first building a 
    `PlateReconstruction` object and associated `PlotTopologies` instance from 
    the provided rotation model, topologies, and static polygons. It then 
    overlays multiple geological datasets—including coastlines, 
    continents, spreading ridges, subduction zones, trench teeth, mineral 
    occurrences, and user-specified grids—on a chosen map projection.

    Parameters
    ----------
    rotation_model : str or pygplates.RotationModel
        Rotation model describing plate kinematics over time.
    topology_features : str or pygplates.FeatureCollection
        Input topological plate boundary features (e.g., polygons, ridges, trenches).
    static_polygons : str or pygplates.FeatureCollection
        Static plate polygons used to anchor reconstructions.
    coastlines : str or geopandas.GeoDataFrame
        Coastline geometry for plotting reconstructed continental outlines.
    continents : str or geopandas.GeoDataFrame
        Present-day continental polygons for plotting reference landmasses.
    COBs : str or geopandas.GeoDataFrame
        Continent–ocean boundaries (optional).
    anchor_plate_id : int
        Anchor plate ID for reconstruction.
    grid : xarray.DataArray or numpy.ndarray
        Primary grid to be visualised (e.g., probability, uncertainty).
    grid_type : {"probability", "uncertainty", "probability_unc", "preservation", "probability_pres", "probability_unc_pres"}
        Type of grid, determines colour scale and legend labels.
    background_grid : xarray.DataArray or numpy.ndarray
        Background raster grid (typically seafloor age) shown beneath primary grid.
    projection : str or cartopy.crs.Projection
        Map projection ("mollweide", "orthographic", or a Cartopy CRS object).
    time : float
        Geological reconstruction time in Ma (millions of years ago).
    positives : str, pd.DataFrame, or None
        Mineral deposit occurrence data (optional), reconstructed if required.
    output_filename : str or Path, optional
        If provided, saves the generated figure to file.
    central_meridian : float
        Central meridian (longitude) for the map projection.
    *_kwargs : dict, optional
        Custom plotting style arguments for each dataset layer, overriding defaults 
        (e.g., `background_kwargs`, `ridges_kwargs`, `scatter_kwargs`).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object containing the plot.

    Notes
    -----
    - Internally constructs a `PlateReconstruction` and `PlotTopologies` object
      to reconstruct features at the given time.
    - Calls `plot()` with the prepared inputs, which handles rendering of 
      coastlines, ridges, trenches, grids, motion vectors, and deposits.
    - Useful for quickly creating consistent global maps in a tectonic 
      reconstruction workflow without manually handling lower-level details.
    """
    
    plate_reconstruction = PlateReconstruction(
        rotation_model=rotation_model,
        topology_features=topology_features,
        static_polygons=static_polygons,
        anchor_plate_id=anchor_plate_id,
    )
    
    gplot = PlotTopologies(
        plate_reconstruction=plate_reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=COBs,
        anchor_plate_id=anchor_plate_id,
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
        uncertainty_kwargs=uncertainty_kwargs,
        probs_unc_kwargs=probs_unc_kwargs,
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
    uncertainty_kwargs: Optional[Mapping[str, Any]] = None,
    probs_unc_kwargs: Optional[Mapping[str, Any]] = None,
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
    
    if grid_type in ("probability", "preservation", "probability_unc", "probability_pres", "probability_unc_pres"):
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
        uncertainty_kwargs=uncertainty_kwargs,
        probs_unc_kwargs=probs_unc_kwargs,
        pres_kwargs=pres_kwargs,
        scatter_kwargs=scatter_kwargs,
        trenches_kwargs=trenches_kwargs,
        teeth_kwargs=teeth_kwargs,
        grids_kwargs=grids_kwargs,
    )
    
    cax_grid = fig.add_axes([0.33, 0.17, 0.25, 0.02])
    cax_bg = fig.add_axes([0.62, 0.17, 0.25, 0.02])
            
    cbar_grid = fig.colorbar(grid_, cax=cax_grid, orientation="horizontal")
    if grid_type == "probability":
        cbar_grid.set_label("Mineralisation probability (%)", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0, 20, 40, 60, 80, 100])
    elif grid_type == "uncertainty":
        cbar_grid.set_label("Uncertainty", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    elif grid_type == "preservation":
        cbar_grid.set_label("Preservation score (%)", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0, 20, 40, 60, 80, 100])
    elif grid_type == "probability_unc":
        cbar_grid.set_label("Adjusted probability by uncertainty (%)", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0, 20, 40, 60, 80, 100])    
    elif grid_type == "probability_pres":
        cbar_grid.set_label("Preserved mineralisation prospectivity (%)", fontsize=FONT_SIZE, labelpad=10)
        cbar_grid.set_ticks([0, 20, 40, 60, 80, 100])    
    elif grid_type == "probability_unc_pres":
        cbar_grid.set_label("Adjusted prospectivity of preserved mineralisation (%)", fontsize=FONT_SIZE-2.5, labelpad=10)
        cbar_grid.set_ticks([0, 20, 40, 60, 80, 100])    

    cbar_grid.ax.tick_params(labelsize=TICK_SIZE)

    cbar_bg = fig.colorbar(bg, cax=cax_bg, orientation="horizontal", extend='max')
    cbar_bg.set_label("Seafloor age (Myr)", fontsize=FONT_SIZE, labelpad=10)
    cbar_bg.set_ticks([0, 50, 100, 150, 200])
    cbar_bg.ax.tick_params(labelsize=TICK_SIZE)
    
    # Dummy handle to trigger custom handler
    trench_handle = Line2D([], [], color="black")
    
    # Add custom handles
    custom_handles = [
        Patch(facecolor=coastlines_kwargs["facecolor"], edgecolor=coastlines_kwargs["edgecolor"], alpha=coastlines_kwargs["alpha"]),
        trench_handle,
        # Line2D([0], [0], color=topologies_kwargs["color"], lw=2),
        Line2D([0], [0], color=ridges_kwargs["color"], lw=2),
        Line2D([0], [0], marker=scatter_kwargs["marker"], markerfacecolor=scatter_kwargs["facecolor"], markeredgecolor=scatter_kwargs["edgecolor"], markersize=15, linestyle='None'),
        ]
    custom_labels = [
        "Continental crust",
        "Subduction zone",
        "Plate boundary",
        "Mineral occurrence",
        ]
    
    # Draw legend
    ax.legend(custom_handles, custom_labels, fontsize=FONT_SIZE, loc="lower left", bbox_to_anchor=(0, -0.22),
              handler_map={trench_handle: HandlerTrenchLine()})
        
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
    anchor_plate_id=None,
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
        anchor_plate_id=anchor_plate_id,
    )
    gplot = PlotTopologies(
        plate_reconstruction=reconstruction,
        coastlines=coastlines,
        continents=continents,
        COBs=COBs,
        anchor_plate_id=anchor_plate_id,
        time=time,
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
    uncertainty_kwargs: Optional[Mapping[str, Any]] = None,
    probs_unc_kwargs: Optional[Mapping[str, Any]] = None,
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
    
    if grid_type == "probability" or grid_type == "probability_unc" or grid_type == "probability_pres" or grid_type == "probability_unc_pres":
        probs_kwargs = {} if probs_kwargs is None else probs_kwargs
        probs_kwargs = _copy_update_dict(PROBS_KWARGS, probs_kwargs)
    elif grid_type == "uncertainty":
        uncertainty_kwargs = {} if uncertainty_kwargs is None else uncertainty_kwargs
        uncertainty_kwargs = _copy_update_dict(UNCERTAINTY_KWARGS, uncertainty_kwargs)
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
    
    # gplot.plot_ridges(
    #     ax=ax,
    #     # central_meridian=central_meridian,
    #     **ridges_kwargs,
    # )
    
    # gplot.plot_transforms(
    #     ax=ax,
    #     # central_meridian=central_meridian,
    #     **ridges_kwargs,
    # )
    
    # gplot.plot_all_topologies(
    #     ax=ax,
    #     # central_meridian=central_meridian,
    #     **topologies_kwargs,
    #     )
    
    gplot.plot_topological_plate_boundaries(
        ax=ax,
        **ridges_kwargs,
    )
    
    gplot.plot_plate_motion_vectors(
        ax=ax,
        **vectors_kwargs,
        )
    
    if grid_type == "probability" or grid_type == "probability_unc" or grid_type == "probability_pres" or grid_type == "probability_unc_pres":
        probs = raster.imshow(
            ax=ax,
            **probs_kwargs,
            )
    elif grid_type == "uncertainty":
        uncertainty = raster.imshow(
            ax=ax,
            **uncertainty_kwargs,
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
            positives = reconstruct_points_(
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

    ax.text(0.505,-0.03, '60°E', transform=ax.transAxes, fontsize=TICK_SIZE)
    ax.text(0.48,-0.03, '0°', transform=ax.transAxes, fontsize=TICK_SIZE)
    ax.text(0.425,-0.027, '60°W', transform=ax.transAxes, fontsize=TICK_SIZE)
    
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

    if grid_type == "probability" or grid_type == "probability_unc" or grid_type == "probability_pres" or grid_type == "probability_unc_pres":
        return bg, probs
    elif grid_type == "uncertainty":
        return bg, uncertainty
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
    
    """
    Create a video animation from a sequence of images using FFmpeg.

    This function takes a list of image file paths and compiles them into a video 
    file. It supports hardware-accelerated encoding (when available), custom 
    codecs, and additional FFmpeg parameters. The output video is written to disk 
    in a format suitable for playback across most video players.

    Parameters
    ----------
    image_filenames : list of str or Path
        List of image file paths in the order they should appear in the video.
    output_filename : str or Path
        Path where the output video will be saved.
    fps : int, default=5
        Frame rate of the input image sequence (frames per second).
    codec : {"auto", "hevc", "libx264", str}, default="auto"
        Video codec to use:
        - "auto": defaults to "libx264"
        - "hevc": uses "hevc_videotoolbox" if hardware acceleration is available, 
          otherwise falls back to "hevc"
        - any other valid FFmpeg codec string.
    bitrate : str, default="5000k"
        Target video bitrate (controls output quality and file size).
    output_fps : int, default=30
        Frame rate of the final rendered video.
    ffmpeg_params : list of str, optional
        Extra FFmpeg command-line parameters (e.g., pixel format, padding).
        Defaults to ensuring dimensions are even and pixel format is "yuv420p".
    **kwargs
        Additional keyword arguments passed to `moviepy.editor.ImageSequenceClip.write_videofile()`,
        such as:
        - `logger` : optional MoviePy logger for progress display.
        - `audio` : bool, whether to include audio track (default False).

    Notes
    -----
    - Uses `moviepy.editor.ImageSequenceClip` internally to create the clip.
    - Ensures that output video dimensions are even (required by many codecs).
    - Supports optional hardware-accelerated HEVC encoding when available 
      (`hevc_videotoolbox` on macOS).
    - By default, sets pixel format to `yuv420p` for maximum compatibility.
    """
    
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


# Custom handler for trench line with triangles
class HandlerTrenchLine(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        # Horizontal line
        line = Line2D([xdescent, xdescent + width], 
                      [ydescent + height / 2] * 2, 
                      color='black', lw=2, transform=trans)

        # Taller, sharper triangle teeth
        def make_triangle(x_center):
            base_width = width * 0.05
            height_offset = height * 0.3
            return Polygon([
                [x_center, ydescent + height / 2 + height_offset],  # tip
                [x_center - base_width, ydescent + height / 2],     # left base
                [x_center + base_width, ydescent + height / 2],     # right base
            ], closed=True, color='black', transform=trans)

        # Two teeth positioned proportionally
        triangle1 = make_triangle(xdescent + width * 0.35)
        triangle2 = make_triangle(xdescent + width * 0.7)

        return [line, triangle1, triangle2]
