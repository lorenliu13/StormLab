# Generate ERA5 covariate fields for each identified rainstorm events
# Yuan Liu
# 2023/01/04

import xarray as xr
import numpy as np
import pandas as pd
import os
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


if __name__ == "__main__":

    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the desired save folder relative to the script's directory
    save_folder = os.path.join(script_dir, '../../../output/era5_rainstorms')

    # Navigate to the desired load folder relative to the script's directory
    era_folder = os.path.join(script_dir, '../../../data/era5/regridded_annual_era5')
    era_tracking_folder = os.path.join(script_dir, '../../../data/era5/era5_tracking')


    year = 2020

    variable_list = ["mean_total_precipitation_rate", "total_column_water_vapour", "vertical_integral_of_water_vapour_flux",
                     "850_u_component_of_wind", "850_v_component_of_wind"]
    variable = "mean_total_precipitation_rate"

    # load the catalog
    ar_catalog = pd.read_csv(
        era_tracking_folder + "/" + "{0}_catalog.csv".format(year))
    ar_ids = np.unique(ar_catalog['storm_id'].values)

    # load data set
    var_xarray = xr.load_dataset(
        era_folder + "/" + "ERA5_6H_{0}_{1}_cesm_res.nc".format(variable, year))
    # get the variable name
    short_name = [k for k in var_xarray.data_vars.keys()]
    short_name = short_name[0]

    for ar_id in ar_ids:
        ar_record = ar_catalog[ar_catalog['storm_id'] == ar_id]

        # convert to datetime and get the day
        storm_timesteps = pd.to_datetime(ar_record['time'])
        # get months
        storm_timestep_months = storm_timesteps.dt.month.values

        # extract the data array
        # select the time during the AR-event
        var_data_xarray = var_xarray.sel(time=ar_record['time'].values)

        # save the era5 with cesm2 resolution
        var_data_xarray.to_netcdf(save_folder + "/" + "{0}_{1}_cesm_res.nc".format(ar_id, variable),
                                  encoding={short_name: {"dtype": "float32", "zlib": True}})

