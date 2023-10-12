# Combine the ERA5 covariate fields of all rainstorm events and regrid them to AORC resolution.
# Yuan Liu
# 2023/01/04


import xarray as xr
import pandas as pd
import numpy as np
import os
from scipy import interpolate


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


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass

variable_list = ["vertical_integral_of_water_vapour_flux", "mean_total_precipitation_rate", "total_column_water_vapour"]
# "mean_total_precipitation_rate", "total_column_water_vapour",
# "mean_vertically_integrated_moisture_divergence",
# "vertical_integral_of_water_vapour_flux",
# "500_vertical_velocity"

# load the corresponding index of aorc data to match the ERA5 grid
# min_dis_index = np.load(r"D:\Miss_design_storm\AORC_data" + "\\" + "min_dis_index_list.npy")
year_list = np.arange(1979, 2022)
# month = 12
month_list = [12, 1, 2, 3, 4, 5]

# set up AORC coordinates
aorc_lat = np.linspace(50, 29, 630)
aorc_lon = np.linspace(-113.16734, -79.068704, 1024)

cesm_lat = np.linspace(50.41884817, 28.7434555, 24)
cesm_lon = np.linspace(-113.75, -78.75, 29)


for variable in variable_list:
    for month in month_list:
        full_monthly_var_list = []
        # create a save folder
        save_directory = r"/home/yliu2232/miss_design_storm/6h_monthly_series/{0}".format(month)
        create_folder(save_directory)

        for year in year_list:
            # if year = 1979 and month = 1
            # skip
            if (year == 1979) & (month == 1):
                continue

            print("Start processing year {0} month {1} variable {2}".format(year, month, variable))
            directory = r"/home/yliu2232/miss_design_storm/6h_tracking_cesm_res/6h_ar_catalog"
            # load csv file
            ar_catalog = pd.read_csv(directory + "/" + "{0}_catalog.csv".format(year))
            # get ar ids
            ar_ids = np.unique(ar_catalog['storm_id'].values)
            # get ar months
            ar_months = ar_catalog.groupby('storm_id').min()['month'].values
            # get ar ids in specific month
            monthly_ar_ids = ar_ids[ar_months == month]

            for ar_id in monthly_ar_ids:
                if ar_id == 202100135:
                    # skip this ar id due to data not available
                    continue
                # ar_record = ar_catalog[ar_catalog['storm_id'] == ar_id]
                # get the ar duration
                # ar_duration = ar_record['duration(hour)'].values[0] * 3
                # only keep those longer than 48 hours
                # if ar_duration < 48:
                #     continue

                # load era5 nc file with cesm-level resolution
                era_folder = r"/home/yliu2232/miss_design_storm/6h_ar_event_cesm_res/{0}/{1}".format(year, ar_id)
                var_xarray = xr.load_dataset(era_folder + "/" + "{0}_{1}_cesm_res.nc".format(ar_id, variable))

                # get the short name
                short_name = [k for k in var_xarray.data_vars.keys()]
                short_name = short_name[0]
                # get the array
                coarse_var_array = var_xarray[short_name].data

                # interpolate to AORC resolution
                scipy_interp_cesm_array = np.zeros((coarse_var_array.shape[0], 630, 1024))

                for time_index in range(coarse_var_array.shape[0]):
                    curr_cesm_array = coarse_var_array[time_index]

                    scipy_interp_cesm_array[time_index] = linear_interpolation(curr_cesm_array, cesm_lon, cesm_lat,
                                                                               aorc_lon, aorc_lat)

                full_monthly_var_list.append(scipy_interp_cesm_array)

        # concatenate
        full_monthly_var_list = np.concatenate(full_monthly_var_list, axis = 0)

        # convert to np.float32
        full_monthly_var_list = full_monthly_var_list.astype(np.float32)

        # save it to directory
        np.save(save_directory + "/" + "{0}_{1}_aorc_res.npy".format(variable, month), full_monthly_var_list)



