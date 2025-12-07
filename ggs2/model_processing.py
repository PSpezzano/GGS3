# author: matthew learn (matt.learn@marine.rutgers.edu)
# This file contains functions for processing model data. Used by 1_main.py
# Contains functions for processing individual model data and comparing model data.

import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import os

from dask.diagnostics import ProgressBar
from datetime import date
from datetime import datetime as dt
import itertools

from .util import (
    print_starttime,
    print_endtime,
    print_runtime,
    generate_data_filename,
    save_data,
)
from .models import CMEMS, ESPC
from .pathfinding import *
from .drift import estimate_segment_drift

"""
Section 1: Individual Model Processing Functions
"""


def regrid_ds(ds1: xr.Dataset, ds2: xr.Dataset, diag_text: bool = True) -> xr.Dataset:
    """
    Regrids the first dataset to the second dataset.

    Args
    ----------
        ds1 (xr.Dataset): The first dataset. This is the dataset that will be regridded.
        ds2 (xr.Dataset): The second dataset. This is the dataset that the first dataset will be regridded to.
        diag_text (bool, optional): Whether to print diagnostic text. Defaults to True.

    Returns
    ----------
        ds1_regridded (xr.Dataset)
            The first dataset regridded to the second dataset.
    """
    text_name = ds1.attrs["text_name"]
    model_name = ds1.attrs["model_name"]
    fname = ds1.attrs["fname"]

    if diag_text:
        print(f"{text_name}: Regridding to {ds2.attrs['text_name']}...")
        starttime = print_starttime()

    # ds1 = ds1.drop_vars(["time"])

    # Code from Mike Smith.
    ds1_regridded = ds1.reindex_like(ds2, method="nearest")

    grid_out = xr.Dataset({"lat": ds2["lat"], "lon": ds2["lon"]})
    regridder = xe.Regridder(ds1, grid_out, "bilinear", extrap_method="nearest_s2d")

    ds1_regridded = regridder(ds1)
    ds1_regridded.attrs["text_name"] = text_name
    ds1_regridded.attrs["model_name"] = model_name
    ds1_regridded.attrs["fname"] = fname

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return ds1_regridded


def interpolate_depth(
    model: object,
    max_depth: int = 1000,
    diag_text: bool = True,
) -> xr.Dataset:
    """
    Interpolates the model data to 1 meter depth intervals.

    Args
    ----------
        model (object): The model data.
        max_depth (int, optional): The maximum depth to interpolate to. Defaults to 1000.
        common_grid (xr.Dataset): The common grid to interpolate to (CMEMS ONLY).
        diag_text (bool, optional): Print diagnostic text. Defaults to True.

    Returns:
    ----------
        ds_interp (xr.Dataset): The interpolated model data.
    """
    ds = model.subset_data

    text_name = ds.attrs["text_name"]
    model_name = ds.attrs["model_name"]

    # Define the depth range that will be interpolated to.
    z_range = np.arange(0, max_depth + 1, 1)

    u = ds["u"]
    v = ds["v"]

    if diag_text:
        print(f"{text_name}: Interpolating depth...")
        starttime = print_starttime()

    u_interp = u.interp(depth=z_range)
    v_interp = v.interp(depth=z_range)

    ds_interp = xr.Dataset({"u": u_interp, "v": v_interp})

    ds_interp.attrs["text_name"] = text_name
    ds_interp.attrs["model_name"] = model_name
    ds_interp.attrs["fname"] = f"{model_name}_zinterp"
    ds_interp = ds_interp.chunk("auto")

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return ds_interp


def depth_average(model: object, diag_text: bool = True) -> xr.Dataset:
    """
    Gets the depth integrated current velocities from the passed model data.

    Args
    ----------
        model (object): The model data.
        diag_text (bool, optional): Whether to print diagnostic text. Defaults to True.

    Returns
    ----------
        ds_da (xr.Dataset):
            The depth averaged model data. Contains 'u', 'v', and 'magnitude' variables.
    """
    ds = model.z_interpolated_data

    text_name = ds.attrs["text_name"]
    model_name = ds.attrs["model_name"]

    if diag_text:
        print(f"{text_name}: Depth averaging...")
        starttime = print_starttime()

    ds_da = ds.mean(dim="depth", keep_attrs=False)

    ds_da.attrs["text_name"] = text_name
    ds_da.attrs["model_name"] = model_name
    ds_da.attrs["fname"] = f"{model_name}_dac"

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return ds_da


def calculate_magnitude(model: object, diag_text: bool = True) -> xr.Dataset:
    """
    Calculates the magnitude of the model data.

    Args
    ----------
        model (object): The model data.
        diag_text (bool, optional): Whether to print diagnostic text. Defaults to True.

    Returns
    ----------
        data_mag (xr.Dataset):
            The model data with a new variable 'magnitude'.
    """
    data = model.da_data

    text_name = data.attrs["text_name"]
    model_name = data.attrs["model_name"]
    fname = data.attrs["fname"]

    if diag_text:
        print(f"{text_name}: Calculating magnitude...")
        starttime = print_starttime()

    # Calculate magnitude (derived from Pythagoras)
    magnitude = np.sqrt(data["u"] ** 2 + data["v"] ** 2)

    data = data.assign(magnitude=magnitude)
    data.attrs["text_name"] = text_name
    data.attrs["model_name"] = model_name
    data.attrs["fname"] = fname
    data = data.chunk("auto")  # just to make sure

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return data

### major test block

def _bearing_AB(latA, lonA, latB, lonB):
    dlon = np.deg2rad(lonB - lonA)
    latA_rad, latB_rad = np.deg2rad(latA), np.deg2rad(latB)
    y = np.sin(dlon) * np.cos(latB_rad)
    x = np.cos(latA_rad)*np.sin(latB_rad) - np.sin(latA_rad)*np.cos(latB_rad)*np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0  # 0°=N, 90°=E

def calculate_along_cross(model: object, bearing_deg: float, name: str = "track", diag_text: bool = True) -> xr.Dataset:
    #Adds {name}_along and {name}_cross to model.da_data from u,v and a compass bearing.
    if diag_text:
        print(f"Calculating {name}_along/{name}_cross...", end=" ")
    data: xr.Dataset = model.da_data

    if "u" not in data or "v" not in data:
        raise ValueError("Expected 'u' and 'v' in depth-averaged data (model.da_data).")

    th = np.deg2rad(bearing_deg)          # 0°=N, 90°=E
    s, c = np.sin(th), np.cos(th)
    # along = u*sin(th) + v*cos(th)  (positive toward bearing)
    along = data["u"] * s + data["v"] * c
    # cross = u*cos(th) - v*sin(th)  (left-positive, 90° CCW from bearing)
    cross = data["u"] * c - data["v"] * s

    data['track_along']=along
    data['track_cross']=cross
    model.track_along=along
    model.track_cross=cross

    model.da_data = data.assign({f"{name}_along": along, f"{name}_cross": cross})
    if diag_text:
        print("Done.")
    return model.da_data
### End block

def calculate_heading(model: object, diag_text: bool = False) -> xr.Dataset:
    """
    Calculates the heading of the model data.

    Args
    ----------
        model (object): The model data object.
        diag_text (bool, optional): Whether to print diagnostic text.

    Returns
    ----------
        data (xr.Dataset)
            The model data with a new variable 'heading'.
    """
    if diag_text:
        print("Calculating heading...", end=" ")

    data = model.da_data
    u = data.u
    v = data.v

    heading = (90 - np.degrees(np.arctan2(v, u))) % 360

    data = data.assign(heading=heading)

    if diag_text:
        print("Done.")

    return data


"""
Section 2: Model Comparison Calculations
"""


def calculate_simple_diff(
    model1: object, model2: object, diag_text: bool = True
) -> xr.Dataset:
    """
    Calculates the simple difference between two datasets. Returns a single xr.Dataset of the simple difference.

    Args
    ----------
        model1 (object): The first model.
        model2 (object): The second model.
        diag_text (bool, optional): Print diagnostic text.


    Returns
    ----------
        simple_diff (xr.Dataset): The simple difference between the two datasets.
    """
    data1: xr.Dataset = model1.da_data
    data2: xr.Dataset = model2.da_data
    model_list: list[xr.Dataset] = [data1, data2]
    model_list.sort(key=lambda x: x.attrs["model_name"])  # sort datasets
    data1 = model_list[0]
    data2 = model_list[1]

    text_name1: str = data1.attrs["text_name"]
    text_name2: str = data2.attrs["text_name"]
    model_name1: str = data1.attrs["model_name"]
    model_name2: str = data2.attrs["model_name"]

    text_name = " & ".join([text_name1, text_name2])
    model_name = "+".join([model_name1, model_name2])

    if diag_text:
        print(f"{text_name}: Calculating Simple Difference...")
        starttime = print_starttime()

    # Calculate speed difference
    speed_diff: xr.DataArray = np.abs(data1.magnitude - data2.magnitude)

    # Calculate heading difference
    heading_diff = np.abs(data1.heading - data2.heading) % 360
    heading_diff = np.minimum(heading_diff, 360 - heading_diff)

    # Merge speed and heading differences
    simple_diff = xr.merge([speed_diff, heading_diff])

    simple_diff.attrs["model_name"] = f"{model_name}_speed_diff"
    simple_diff.attrs["text_name"] = f"Simple Difference [{text_name}]"
    simple_diff.attrs["fname"] = simple_diff.attrs["model_name"]
    # vvvvv not sure if this is going to do what I want it to
    simple_diff.attrs["model1_name"] = text_name1
    simple_diff.attrs["model2_name"] = text_name2

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return simple_diff


def calculate_percent_diff(
    model1: object, model2: object, eps: float = 1e-6, diag_text: bool = True
) -> xr.Dataset:
    """
    Compute the pointwise percent difference between two fields:
        100 * |a - b| / ((|a| + |b|)/2 + eps)

    Args
    ----------
        model1 (object): The first model.
        model2 (object): The second model.
        eps (float): epsilon, a small number to avoid division by zero. Defaults to 1e-6

    Returns
    ----------
        percent_diff (xr.Dataset)
            The percent difference
    """
    if diag_text:
        print("Calculating percent difference...", end=" ")

    data1: xr.Dataset = model1.da_data
    data2: xr.Dataset = model2.da_data
    model_list: list[xr.Dataset] = [data1, data2]
    model_list.sort(key=lambda x: x.attrs["model_name"])  # sort datasets
    data1 = model_list[0]
    data2 = model_list[1]

    text_name1: str = data1.attrs["text_name"]
    text_name2: str = data2.attrs["text_name"]
    model_name1: str = data1.attrs["model_name"]
    model_name2: str = data2.attrs["model_name"]

    text_name = " & ".join([text_name1, text_name2])
    model_name = "+".join([model_name1, model_name2])

    # Calculate percent difference
    num = np.abs(data1 - data2)
    denom = (np.abs(data1) + np.abs(data2)) / 2 + eps
    perc_diff = 100 * num / denom

    perc_diff.attrs["model_name"] = f"{model_name}_percent_diff"
    perc_diff.attrs["text_name"] = f"Percent Difference [{text_name}]"
    perc_diff.attrs["fname"] = perc_diff.attrs["model_name"]
    # vvvvv not sure if this is going to do what I want it to
    perc_diff.attrs["model1_name"] = text_name1
    perc_diff.attrs["model2_name"] = text_name2

    if diag_text:
        print("Done.")

    return perc_diff


def calculate_vector_diff(
    model1: object, model2: object, eps: float = 1e-6, diag_text: bool = True
) -> xr.DataArray:
    """
    Calculate the vector percent difference between two datasets. Accounts for both heading and speed differences.

    Args
    ----------
        model1 (object): The first model.
        model2 (object): The second model.
        eps (float): epsilon, a small number to avoid division by zero. Defaults to 1e-6
        diag_text (bool, optional): Whether to print diagnostic text.

    Returns
    ----------
        vector_diff (xr.DataArray)
            The vector difference
    """
    if diag_text:
        print("Calculating vector difference...", end=" ")

    data1: xr.Dataset = model1.da_data
    data2: xr.Dataset = model2.da_data
    model_list: list[xr.Dataset] = [data1, data2]
    model_list.sort(key=lambda x: x.attrs["model_name"])  # sort datasets
    data1 = model_list[0]
    data2 = model_list[1]

    text_name1: str = data1.attrs["text_name"]
    text_name2: str = data2.attrs["text_name"]
    model_name1: str = data1.attrs["model_name"]
    model_name2: str = data2.attrs["model_name"]

    text_name = " & ".join([text_name1, text_name2])
    model_name = "+".join([model_name1, model_name2])

    # Calculate vector percent difference
    u1 = data1.u
    v1 = data1.v
    u2 = data2.u
    v2 = data2.v
    diff = np.sqrt((u1 - u2) ** 2 + (v1 - v2) ** 2)
    mag1 = np.sqrt(u1**2 + v1**2)
    mag2 = np.sqrt(u2**2 + v2**2)
    mean_mag = (mag1 + mag2) / 2 + eps
    vector_diff = 100 * diff / mean_mag

    vector_diff.attrs["model_name"] = f"{model_name}_vector_diff"
    vector_diff.attrs["text_name"] = f"Vector Difference [{text_name}]"
    vector_diff.attrs["fname"] = vector_diff.attrs["model_name"]
    vector_diff.attrs["model1_name"] = text_name1
    vector_diff.attrs["model2_name"] = text_name2

    if diag_text:
        print("Done.")

    return vector_diff


def calculate_rms_vertical_diff(
    model1: object, model2: object, regrid: bool = False, diag_text: bool = True
) -> xr.Dataset:
    """
    Calculates the vertical root mean squared difference between two datasets.

    Args
    ----------
        model1 (object): The first model.
        model2 (object): The second model.
        regrid (bool, optional): Whether to regrid datasets. Defaults to `False`.\
        diag_text (bool, optional): Print diagnostic text.

    Returns:
    ----------
        vrmsd (xr.Dataset): The vertical root mean squared difference between the two datasets.
    """
    data1 = model1.z_interpolated_data
    data2 = model2.z_interpolated_data
    model_list = [data1, data2]
    model_list.sort(key=lambda x: x.attrs["model_name"])  # sort datasets
    data1 = model_list[0]
    data2 = model_list[1]

    text_name1: str = data1.attrs["text_name"]
    text_name2: str = data2.attrs["text_name"]
    model_name1: str = data1.attrs["model_name"]
    model_name2: str = data2.attrs["model_name"]

    text_name = " & ".join([text_name1, text_name2])
    model_name = "+".join([model_name1, model_name2])

    if diag_text:
        print(f"{text_name}: Calculating RMSD...")
        starttime = print_starttime()

    if regrid:
        data2 = regrid_ds(data2, data1, diag_text=False)  # regrid model2 to model1.

    # Calculate RMSD
    delta_u = data1.u - data2.u
    delta_v = data1.v - data2.v

    vrmsd_u = np.sqrt(np.square(delta_u).mean(dim="depth"))
    vrmsd_v = np.sqrt(np.square(delta_v).mean(dim="depth"))
    vrmsd_mag = np.sqrt(vrmsd_u**2 + vrmsd_v**2)

    vrmsd = xr.Dataset(
        {"u": vrmsd_u, "v": vrmsd_v, "magnitude": vrmsd_mag},
        attrs={"text_name": text_name, "model_name": model_name, "fname": model_name},
    )

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return vrmsd


def calculate_simple_mean(
    model_list: list[object], diag_text: bool = True
) -> xr.Dataset:
    """
    Calculates the simple mean of a list of datasets. Returns a single xr.Dataset of the simple means.

    Args
    ----------
        model_list (list[object]): A list of xr.Datasets.
        diag_text (bool, optional): Print diagnostic text.

    Returns
    ----------
        simple_mean (xr.Dataset): The simple mean of the list of datasets.
    """
    if diag_text:
        print("Calculating simple mean of selected models...")
        starttime = print_starttime()

    datasets = [model.da_data for model in model_list]
    model_names = "_".join([dataset.attrs["model_name"] for dataset in datasets])
    text_names = ", ".join([dataset.attrs["text_name"] for dataset in datasets])

    combined_dataset = xr.concat(datasets, dim="datasets")
    simple_mean = combined_dataset.mean(dim="datasets")
    simple_mean.attrs["model_name"] = f"{model_names}_simple_mean"
    simple_mean.attrs["text_name"] = f"Simple Mean [{text_names}]"
    simple_mean.attrs["fname"] = simple_mean.attrs["model_name"]

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return simple_mean


def calculate_mean_diff(model_list: list[object], diag_text: bool = True) -> xr.Dataset:
    """
    Calculates the mean of the differences of each non-repeating pair of models from the passed list of datasets.
    Returns a single xr.Dataset of the mean differences.

    Args
    ----------
        model_list (list[object]): A list of xr.Datasets.
        diag_text (bool, optional): Print diagnostic text.

    Returns
    ----------
        mean_diff (xr.Dataset): The mean difference of all selected models.
    """
    if diag_text:
        print("Calculating mean difference of selected models...")
        starttime = print_starttime()

    datasets = [model.da_data for model in model_list]
    model_names = "_".join([dataset.attrs["model_name"] for dataset in datasets])
    text_names = ", ".join([dataset.attrs["text_name"] for dataset in datasets])

    ds_combos = list(itertools.combinations(datasets, r=2))

    diff_list = []
    for ds1, ds2 in ds_combos:
        diff_list.append(abs(ds1 - ds2))

    combined_ds = xr.concat(diff_list, dim="datasets", coords="minimal")
    mean_diff = combined_ds.mean(dim="datasets")
    mean_diff.attrs["model_name"] = f"{model_names}_meandiff"
    mean_diff.attrs["text_name"] = f"Mean Difference [{text_names}]"
    mean_diff.attrs["fname"] = mean_diff.attrs["model_name"]

    if diag_text:
        print("Done.")
        endtime = print_endtime()
        print_runtime(starttime, endtime)

    return mean_diff


"""
Section 3: Model Processing Functions
"""


def process_common_grid(
    dates: tuple[str, str], extent: tuple[float, float, float, float], depth: int
) -> xr.Dataset:
    """
    Loads and subsets data from ESPC to act as a common grid for all models. In the event of a failure, CMEMS will be used as a common grid instead.

    Args
    ----------
        dates (tuple[str, str]): A tuple of (start_date, end_date).
        extent (tuple[float, float, float, float]): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        depth (int): The maximum depth in meters.

    Returns
    ----------
        common_grid (xr.Dataset): Common grid data.
    """
    print("Setting up COMMON_GRID...")
    starttime = print_starttime()

    try:
        temp = ESPC()
        temp.load(diag_text=False)
        temp.raw_data.attrs["text_name"] = "COMMON GRID"
        temp.raw_data.attrs["model_name"] = "COMMON_GRID"
        today = date.today()
        temp.subset((today, today), extent, depth, diag_text=False)
        temp.subset_data["time"] = [np.datetime64(dates[0])]
        temp.subset_data = temp.subset_data.isel(time=0)
        common_grid = temp.subset_data
    except Exception as e:
        print(f"ERROR: Failed to process ESPC COMMON GRID data due to: {e}\n")
        print("Processing CMEMS COMMON GRID instead...\n")
        temp = CMEMS()
        temp.load(diag_text=False)
        temp.raw_data.attrs["text_name"] = "COMMON GRID"
        temp.raw_data.attrs["model_name"] = "COMMON_GRID"
        temp.subset(dates, extent, depth, diag_text=False)
        common_grid = temp.subset_data

    print("Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)

    return common_grid


def process_individual_model(
    model: object,
    common_grid: xr.Dataset,
    dates: tuple[str, str],
    extent: tuple[float, float, float, float],
    depth: int,
    single_date: bool = True,
    pathfinding: bool = False,
    heuristic: str = None,
    waypoints: list[tuple[float, float]] = None,
    glider_speed: float = None,
    mission_name: str = None,
    save: bool = False,
) -> None:
    
    dir='products/2025_08_08_pathfinding_output/data'
    os.makedirs(dir,exist_ok=True)
    if glider_speed is None:
        glider_speed=0.40 
    """
    Processes individual model data. Assigns regridded subset data,
    1 meter interval interpolated data, & depth averaged to model class attributes.

    Args
    ----------
        model (object): The model data.
        common_grid (xr.Dataset): Common grid data.
        dates (tuple[str, str]): A tuple of (date_min, date_max) in datetime format.
        extent (tuple[float, float, float, float]): A tuple of (lat_min, lon_min, lat_max, lon_max) in decimel degrees.
        depth (int): The maximum depth in meters.
        single_date (bool): Boolean indicating whether to subset data to a single datetime.
        pathfinding (dict): Dictionary of pathfinding parameters.
        heuristic (str): Pathfinding heuristic. Options: "drift_aware", "haversine".
        waypoints (list[tuple[float, float]]): List of waypoints for the A* computation.
        mission_name (str): Name of the mission.
        save (bool): Save each data to netCDF.
    """
    print(f"{model.name}: Processing data...")
    starttime = print_starttime()

    # subset
    model.subset(dates, extent, depth, diag_text=False)

    if "time" in model.subset_data.dims:
        # Select first timestep, drop time as dimension
        model.subset_data = model.subset_data.isel(time=0).drop_vars("time")
        # Add as coordinate
        model.subset_data = model.subset_data.assign_coords(
            time=("time", [np.datetime64(dates[0])])
        )
    model.subset_data = regrid_ds(model.subset_data, common_grid, diag_text=False)

    # interpolate depth
    model.z_interpolated_data = interpolate_depth(model, depth, diag_text=False)
    with ProgressBar(minimum=1):
        model.z_interpolated_data = model.z_interpolated_data.persist()

    # depth average
    model.da_data = depth_average(model, diag_text=False)
    model.da_data = calculate_magnitude(model, diag_text=False)
    model.da_data = calculate_heading(model, diag_text=False)
## second test block
    if hasattr(model, "waypoints") and model.waypoints and len(model.waypoints) >= 2:
        latA, lonA = model.waypoints[0]
        latB, lonB = model.waypoints[-1]   # overall mission direction
        bearing = _bearing_AB(latA, lonA, latB, lonB)
    else:
        bearing = 90.0  # default if no waypoints (0°=N, 90°=E)

    model.da_data = calculate_along_cross(
        model,
        bearing_deg=bearing,
        name="track",
        diag_text=True,
)

    with ProgressBar(minimum=1):
     # TODO: would it be faster for A* to be computed with it loaded?
        model.da_data = model.da_data.compute()
    if save:
        fdate = "2025111500"
        ddate = "2025_11_15"
        fname_zi = model.z_interpolated_data.attrs["fname"]
        fname_da = model.da_data.attrs["fname"]

        full_fname_zi = generate_data_filename(mission_name, fdate, fname_zi)
        full_fname_da = generate_data_filename(mission_name, fdate, fname_da)

        #save_data(model.z_interpolated_data, full_fname_zi, ddate)
        save_data(model.da_data, full_fname_da, ddate)

    print("Processing Done.")
    endtime = print_endtime()
    print_runtime(starttime, endtime)

    # pathfinding
    if pathfinding:
        ds=model.da_data
        u_perp_array=ds['track_cross'].values
        lambda_weight=10
        model.waypoints = waypoints
        model.optimal_path = compute_a_star_path(
            waypoints,
            model,
            heuristic,
            glider_speed,
            mission_name,
            u_perp_array=u_perp_array,
            lambda_weight=lambda_weight,
        )
    else:
        model.waypoints = None
        model.optimal_path = None
# connection point 1 of drift 
    if getattr(model,'optimal_path',None) is not None:

        path_lat=np.array([p[0] for p in model.optimal_path])
        path_lon=np.array([p[1] for p in model.optimal_path])
        print ('first few points') # print test
        for j in range (min(3,len(path_lon))):
            print (j,path_lon[j],path_lat[j])
        drift_records=[]
        cumul_drift=0.0 # dist in km
    
        for i in range(len(path_lon)-1):
            lonA,latA=float(path_lon[i]), float(path_lat[i])
            lonB,latB=float(path_lon[i+1]),float(path_lat[i+1])

            T_hours, drift_km,u_par_mid,u_perp_mid, R=estimate_segment_drift(
                lonA,latA,lonB,latB,
                model.track_along,
                model.track_cross,
                glider_speed,
            )
            #cumul_drift+=drift_km if np.isfinite(drift_km) else 0.0
            if np.isfinite(drift_km):
                cumul_drift+=drift_km

            lon_mid=0.5*(lonA+lonB)
            lat_mid=0.5*(latA+latB)

            drift_records.append(
                {
                    'seg_index': i,
                    'lon_mid': lon_mid,
                    'lat_mid': lat_mid,
                    'T_hours': T_hours,
                    'drift_km': drift_km,
                    'u_perp_mps':u_perp_mid,
                    'u_par_mps': u_par_mid,
                    'cumul_drift_km': cumul_drift,
                    'R': R,
                
                }
            )

        print("Phase 2 drift: segments =", len(drift_records))
        model.drift_records = drift_records
        out_dir='products/2025_08_08_pathfinding_output_data'
        os.makedirs(out_dir,exist_ok=True)

        drift_df=pd.DataFrame(drift_records)
        drift_csv_path=os.path.join(
            out_dir,
            f'{mission_name}_drift_summary.csv',
        )

        drift_df.to_csv(drift_csv_path,index=False)
        print(f'[DRIFT] Saved drift summary to {drift_csv_path}')
    
    else:
        print("DEBUG: no optimal_path, skipping drift")
    return model


 