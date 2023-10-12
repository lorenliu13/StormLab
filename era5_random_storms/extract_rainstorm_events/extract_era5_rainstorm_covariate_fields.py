# Generate ERA5 covariate fields for each identified rainstorm events
# Yuan Liu
# 2023/01/04

import xarray as xr
import numpy as np
import xesmf as xe
import pandas as pd
import os
import multiprocessing
from scipy import interpolate


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


def linear_interpolation(array, old_x, old_y, new_x, new_y):
    """
    Linear interpolation of current 2-d field. Note that the latitude need to be flipped.
    :param array:
    :param old_x:
    :param old_y:
    :param new_x:
    :param new_y:
    :return:
    """
    # get current array
    curr_flip_array = np.flip(array, axis=0)

    # create interpolation function
    f = interpolate.interp2d(old_x, np.flip(old_y), curr_flip_array, kind = 'linear')

    # use it to interpolate to new grid
    new_z = f(new_x, np.flip(new_y))

    # flip it
    new_z = np.flip(new_z, axis = 0)

    return new_z


def extract_ar_era(year, variable):

    print("Start processing {0} {1}".format(year, variable))
    # load the catalog
    ar_catalog = pd.read_csv(r"/home/yliu2232/miss_design_storm/6h_tracking_cesm_res/6h_ar_catalog" + "/" + "{0}_catalog.csv".format(year))
    ar_ids = np.unique(ar_catalog['storm_id'].values)

    # load data set
    var_xarray = xr.load_dataset(
        "/home/yliu2232/miss_design_storm/raw_data/ERA5/ERA5_6h" + "/" + str(year) + "/" + "ERA5_6H_{0}_{1}_cesm_res.nc".format(variable, year))
    # get the variable name
    short_name = [k for k in var_xarray.data_vars.keys()]
    short_name = short_name[0]

    for ar_id in ar_ids:
        ar_record = ar_catalog[ar_catalog['storm_id'] == ar_id]

        # convert to datetime and get the day
        storm_timesteps = pd.to_datetime(ar_record['time'])
        # get months
        storm_timestep_months = storm_timesteps.dt.month.values

        # if in year 1979, skip storm that starts in January
        if (year == 1979) and (storm_timestep_months[0] == 1):
            print("AORC data is not available for storm {0}".format(ar_id))
            continue

        # if ar_id == 202100112:
        #     # skip this storm event because missing aorc data at 20211231
        #     continue

        # extract the data array
        # select the time during the AR-event
        var_data_xarray = var_xarray.sel(time=ar_record['time'].values)

        # create a folder
        save_folder = r"/home/yliu2232/miss_design_storm/6h_ar_event_cesm_res/{0}/{1}".format(year, ar_id)
        create_folder(save_folder)

        # save the era5 with cesm2 resolution
        var_data_xarray.to_netcdf(save_folder + "/" + "{0}_{1}_cesm_res.nc".format(ar_id, variable), encoding={short_name: {"dtype": "float32", "zlib": True}})


def run_task(task):

    year = task['year']
    variable = task['variable']

    extract_ar_era(year, variable)

if __name__ == "__main__":

    # year_list = np.arange(1979, 2011)
    year_list = np.arange(1979, 2022)
    variable_dict = {}
    # variable_list = ["mean_total_precipitation_rate", "total_column_water_vapour",
    #                        "mean_vertically_integrated_moisture_divergence",
    #                        "vertical_integral_of_water_vapour_flux",
    #                        "500_vertical_velocity"]

    # Run on 02/11/2023, attach 850 mb wind to each AR event
    variable_list = ["mean_total_precipitation_rate", "total_column_water_vapour", "vertical_integral_of_water_vapour_flux",
                     "850_u_component_of_wind", "850_v_component_of_wind"]
    # variable_list = ["vertical_integral_of_water_vapour_flux"]

    task_list = []
    for variable in variable_list:
        for year in year_list:
            task = {'year':year, 'variable':variable}
            task_list.append(task)
            # file_folder = r"/slow/yliu2232/ERA5/ERA5_3h"
            # extract_ar_era(year, file_folder, variable)
            # print("Processing {0} in {1}".format(variable, year))

    pool = multiprocessing.Pool(processes=6)
    pool.map(run_task, task_list)
    print("All task is finished.")
