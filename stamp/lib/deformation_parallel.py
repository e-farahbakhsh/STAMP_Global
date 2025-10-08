import warnings
from pathlib import Path
from joblib import Parallel, delayed, cpu_count

import numpy as np
import pandas as pd
import pygplates
from gplately import PlateReconstruction
from sklearn.neighbors import RadiusNeighborsRegressor


def extract_strain_and_rate(
    data,
    topological_model,
    max_time,
    dropna=False,
    age_col="age (Ma)",
    n_jobs=1,
):
    if not isinstance(data, pd.DataFrame):
        try:
            if Path(str(data)).is_file():
                data = pd.read_csv(data)
        except Exception:
            pass
        else:
            data = pd.DataFrame(data)

    if age_col not in data.columns:
        raise ValueError(f"Column '{age_col}' not found in data: {data.columns}")

    if isinstance(topological_model, PlateReconstruction):
        topo_model_args = dict(
            topological_features=topological_model.topology_features,
            rotation_model=topological_model.rotation_model,
        )
    else:
        topo_model_args = dict(topological_model)

    grouped = list(data.groupby(age_col))

    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    elif n_jobs < 0:
        n_jobs = cpu_count() + n_jobs + 1

    def process_time_group(group):
        time, subset = group
        time = np.around(time)

        model = pygplates.TopologicalModel(**topo_model_args)
        time_span = _get_time_span(model, max_time, resolution=0.5)

        result = pd.DataFrame(index=subset.index, columns=[
            "dilatation_strain_rate (/Ps)",
            "shear_strain_rate (rad/Ps)",
            "strain_style",
            "total_strain_rate (/Ps)",
            "dilatation_strain",
        ])

        snapshot = model.topological_snapshot(time)
        networks = snapshot.get_resolved_topologies(pygplates.ResolveTopologyType.network)
        points = [pygplates.PointOnSphere(lat, lon) for lat, lon in zip(subset["lat"], subset["lon"])]

        for index, point in zip(subset.index, points):
            for network in networks:
                geom = network.get_resolved_geometry()
                if geom.is_point_in_polygon(point):
                    strain_rate = network.get_point_strain_rate(point)
                    if strain_rate is None:
                        strain_rate = pygplates.StrainRate.zero
                    result.at[index, "dilatation_strain_rate (/Ps)"] = strain_rate.get_dilatation_rate() * 1.0e15
                    result.at[index, "shear_strain_rate (rad/Ps)"] = _max_shear_rate(strain_rate.get_rate_of_deformation()) * 1.0e15
                    result.at[index, "strain_style"] = np.clip(strain_rate.get_strain_rate_style(), -1.0, 1.0)
                    result.at[index, "total_strain_rate (/Ps)"] = strain_rate.get_total_strain_rate() * 1.0e15
                    break

        strain_points = time_span.get_geometry_points(time)
        strains = time_span.get_strains(time)

        point_lons, point_lats, point_strains = [], [], []
        for p, s in zip(strain_points, strains):
            if s is None:
                continue
            d = s.get_dilatation()
            if d != 0.0:
                lat, lon = p.to_lat_lon()
                point_lons.append(lon)
                point_lats.append(lat)
                point_strains.append(d)

        if point_lons:
            x_train = pygplates.MultiPointOnSphere(np.column_stack((point_lats, point_lons))).to_xyz_array()
            x_pred = pygplates.MultiPointOnSphere(points).to_xyz_array()
            neigh = RadiusNeighborsRegressor(radius=np.sqrt(2 * (1 - np.cos(np.deg2rad(0.5)))))
            neigh.fit(x_train, np.array(point_strains))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                cumulative_strains = neigh.predict(x_pred)
            for index, strain in zip(subset.index, cumulative_strains):
                result.at[index, "dilatation_strain"] = strain

        return result

    results = Parallel(n_jobs=n_jobs)(delayed(process_time_group)(group) for group in grouped)
    strain_df = pd.concat(results)
    out = pd.concat((data, strain_df), axis="columns")

    if dropna:
        out = out.dropna(subset=[
            "dilatation_strain_rate (/Ps)",
            "shear_strain_rate (rad/Ps)",
            "strain_style",
            "total_strain_rate (/Ps)",
            "dilatation_strain",
        ])
    return out


def extract_strain_history(
    data,
    topological_model,
    time_window=10,
    dropna=False,
    age_col="age (Ma)",
    columns=None,
    n_jobs=1,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        try:
            if Path(str(data)).is_file():
                data = pd.read_csv(data)
        except Exception:
            pass
        else:
            data = pd.DataFrame(data)

    if isinstance(topological_model, PlateReconstruction):
        topo_model_args = dict(
            topological_features=topological_model.topology_features,
            rotation_model=topological_model.rotation_model,
        )
    else:
        topo_model_args = dict(topological_model)

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
        data[_diff_column(column)] = np.nan
        data[_mean_column(column)] = np.nan
        new_columns.extend([_diff_column(column), _mean_column(column)])

    dts = range(time_window)
    grouped = list(data.groupby(data[age_col].round()))

    def process_group(group):
        time, subset = group
        model = pygplates.TopologicalModel(**topo_model_args)
        initial_points = pygplates.MultiPointOnSphere(np.column_stack((subset["lat"], subset["lon"])))
        time_span = model.reconstruct_geometry(initial_points, initial_time=time, oldest_time=time + time_window - 1)

        values = {col: pd.DataFrame(index=subset.index, columns=dts) for col in columns}

        for dt in dts:
            current_time = time + dt
            for i, sr in zip(subset.index, time_span.get_strain_rates(current_time)):
                if sr is None:
                    continue
                values["shear_strain_rate (rad/Ps)"].at[i, dt] = _max_shear_rate(sr.get_rate_of_deformation()) * 1.0e15
                values["dilatation_strain_rate (/Ps)"].at[i, dt] = sr.get_dilatation_rate() * 1.0e15
                values["total_strain_rate (/Ps)"].at[i, dt] = sr.get_total_strain_rate() * 1.0e15

        result = {}
        for column in columns:
            diff_col = _diff_column(column)
            mean_col = _mean_column(column)
            for i, row in values[column].iterrows():
                row = row.dropna()
                if len(row) > 1:
                    result.setdefault(diff_col, {})[i] = (row.iloc[0] - row.iloc[-1]) / (row.index[-1] - row.index[0])
                    result.setdefault(mean_col, {})[i] = row.mean()
        return result

    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    elif n_jobs < 0:
        n_jobs = cpu_count() + n_jobs + 1

    results = Parallel(n_jobs=n_jobs)(delayed(process_group)(group) for group in grouped)

    # Merge all results into original data
    for res in results:
        for col, values in res.items():
            for idx, val in values.items():
                data.at[idx, col] = val

    if dropna:
        data = data.dropna(subset=new_columns)

    return data


def _get_time_span(topological_model, max_time, resolution=0.5):
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
        max_time,
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