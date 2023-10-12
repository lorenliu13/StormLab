# Generate CESM2 covariate fields for each identified rainstorm events.
# Yuan Liu
# 2023/01/04

import xarray as xr
import numpy as np
import xesmf as xe
import pandas as pd
import os
import multiprocessing
from datetime import datetime


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass

def extract_ar_era(ensemble_year, ensemble_id, year, month, variable):

    print("Process ensemble_year {0} ensemble_id {1} year {2} month {3} variable {4}".format(ensemble_year, ensemble_id, year, month, variable))

    # load regridder to AORC resolution

    # load the catalog
    ar_catalog = pd.read_csv(r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_catalog_bs/{0}_{1}".format(ensemble_year, ensemble_id) + "/" + "{0}_catalog.csv".format(year))
    ar_ids = np.unique(ar_catalog['storm_id'].values)

    # get ar months
    ar_months = ar_catalog.groupby('storm_id').min()['month'].values
    # get ar ids in specific month
    monthly_ar_ids = ar_ids[ar_months == month]

    if (variable == 'prect') | (variable == 'tmq') | (variable == 'ivt'):
        # load from bias-corrected data
        var_xarray = xr.load_dataset(
            "/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/bias_corrected_series/{0}_{1}".format(ensemble_year,
                                                                                          ensemble_id) + "/" + str(
                year) + "/" + "CESM2_{0}_{1}_bs.nc".format(year, variable))
    else: # for wind speed
        # load data set
        var_xarray = xr.load_dataset(
            "/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour/{0}_{1}".format(ensemble_year, ensemble_id) + "/" + str(year) + "/" + "CESM2_{0}_{1}.nc".format(variable, year))

    # get the variable name
    short_name = [k for k in var_xarray.data_vars.keys()]
    short_name = short_name[0]

    # get the time step of full-year cesm data
    full_time_steps = var_xarray['time'].data
    # convert ctftime to datetime objects
    full_time_steps = [datetime(item.year, item.month, item.day, item.hour, item.minute, item.second) for item in
                       full_time_steps]

    # get the data array
    var_array = var_xarray[short_name].data
    # get raw lat and lon
    raw_cesm_lat_data = var_xarray['latitude'].data
    raw_cesm_lon_data = var_xarray['longitude'].data


    for ar_id in monthly_ar_ids:
        ar_record = ar_catalog[ar_catalog['storm_id'] == ar_id]

        # get ar time steps
        ar_time = ar_record['time'].values
        # convert strings from array A to datetime objects
        ar_time_datetime = [datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S") for date_str in ar_time]

        # find the positions of the elements of array_A_datetime in array_B
        positions = [i for i, date in enumerate(full_time_steps) if date in ar_time_datetime]
        positions = np.array(positions)

        if positions.shape[0] != ar_time.shape[0]:
                print('Mismatch between ar time.')
                # continue

        # get the period of ar-event
        ar_var_array = var_array[positions]

        # create a data xarray based on subset data
        ar_var_ds = xr.Dataset(
            {short_name: (['time', 'latitude', 'longitude'], ar_var_array)},
            coords={
            'time': ar_time,
            'latitude': raw_cesm_lat_data,
            'longitude': raw_cesm_lon_data
        },
        attrs={'description': "CESM2 {0} data over the Mississippi River Basin for ar event {1}".format(short_name,
            ar_id)}
        )

        # regrid it to cesm2 resolution
        subset_1024_lat = np.linspace(50, 29, 630)
        subset_1024_lon = np.linspace(-113.16734, -79.068704, 1024)
        # target_grid = xr.Dataset({
        #     'latitude': (('latitude',), subset_1024_lat),
        #     'longitude': (('longitude',), subset_1024_lon)
        # })
        # # set up regridder
        # regridder = xe.Regridder(ar_var_ds, target_grid, method='bilinear')
        # # regrid the cesm-level array to high-resolution
        # aorc_res_xarray = regridder(ar_var_ds, keep_attrs=True)

        # create a folder
        save_folder = r"/home/yliu2232/miss_design_storm/cesm2_random_storms/cesm2_ar_covariate_field_bs/{0}_{1}/{2}/{3}".format(ensemble_year, ensemble_id, year, ar_id)
        create_folder(save_folder)
        # save the file with variable name
        ar_var_ds.to_netcdf(save_folder + "/" + "{0}_{1}_cesm_res.nc".format(ar_id, variable))
        # aorc_res_xarray.to_netcdf(save_folder + "/" + "{0}_{1}_aorc_res.nc".format(ar_id, variable))



def run_task(task):

    year = task['year']
    variable = task['variable']
    ensemble_year = task['ensemble_year']
    ensemble_id = task['ensemble_id']
    month = task['month']

    extract_ar_era(ensemble_year, ensemble_id, year, month, variable)

if __name__ == "__main__":

    ensemble_year = 1251
    # ensemble_id = 12
    ensemble_ids = np.arange(16, 21)
    # ensemble_ids = [11]
    # month = 12
    month_list = [1, 2, 3, 4, 5, 12]

    year_list = np.arange(1950, 2051)
    variable_dict = {}
    # variable_list = ["mean_total_precipitation_rate", "total_column_water_vapour",
    #                        "mean_vertically_integrated_moisture_divergence",
    #                        "vertical_integral_of_water_vapour_flux",
    #                        "500_vertical_velocity"]

    # Run on 02/11/2023, attach 850 mb wind to each AR event
    variable_list = ["tmq", "prect", 'u850', 'v850', 'ivt']
    # variable_list = ['ivt']

    # variable_list = ['prect', 'tmq']

    task_list = []
    for ensemble_id in ensemble_ids:
        for month in month_list:
            for variable in variable_list:
                for year in year_list:
                    task = {'ensemble_id':ensemble_id, 'ensemble_year':ensemble_year, 'year':year, 'month':month, 'variable':variable}
                    task_list.append(task)
                    # file_folder = r"/slow/yliu2232/ERA5/ERA5_3h"
                    # extract_ar_era(year, file_folder, variable)
                    # print("Processing {0} in {1}".format(variable, year))

    pool = multiprocessing.Pool(processes=6)
    pool.map(run_task, task_list)
    print("All task is finished.")
