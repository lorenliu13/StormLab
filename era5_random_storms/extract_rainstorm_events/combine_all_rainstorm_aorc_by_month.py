# Combine the AORC rainfall fields of all rainstorm events.
# Yuan Liu
# 2023/02/08


import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


# load the corresponding index of aorc data to match the ERA5 grid
# min_dis_index = np.load(r"D:\Miss_design_storm\AORC_data" + "\\" + "min_dis_index_list.npy")
year_list = np.arange(1979, 2022)
# month = 12
# month_list = np.arange(1, 13)
# month_list = [12]
month_list = [12, 1, 2, 3, 4, 5]

for month in month_list:
    full_monthly_aorc_rainfall_list = []
    # create a save folder
    save_directory = r"/home/yliu2232/miss_design_storm/6h_monthly_series/{0}".format(month)
    create_folder(save_directory)

    for year in year_list:
        # if year is in 1979, skip the first month
        if (year == 1979) & (month == 1):
            continue
        print("Start processing year {0} month {1}".format(year, month))
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

            # if ar_id == 202100315:
            #     # skip this ar id due to data not available
            #     continue
            # ar_record = ar_catalog[ar_catalog['storm_id'] == ar_id]
            # get the ar duration
            # ar_duration = ar_record['duration(hour)'].values[0] * 3
            # only keep those longer than 48 hours
            # if ar_duration < 48:
            #     continue

            # load ar nc file
            aorc_xarray_loc = r"/home/yliu2232/miss_design_storm/6h_ar_event_cesm_res/{0}/{1}".format(year, ar_id)
            aorc_xarray = xr.load_dataset(aorc_xarray_loc + "/" + "{0}_aorc.nc".format(ar_id))
            aorc_array = aorc_xarray['aorc'].data

            # flatten the array into (24, lon_grids * lat_grids)
            # aorc_array_flatten = aorc_array.reshape(aorc_array.shape[0], aorc_array.shape[1] * aorc_array.shape[2])
            # # extract at ERA data points
            # era_grid_aorc_array = aorc_array_flatten[:, min_dis_index]
            # # reshape to ERA5 grid space
            # era_grid_aorc_array = era_grid_aorc_array.reshape(aorc_array.shape[0], 85, 153)

            full_monthly_aorc_rainfall_list.append(aorc_array)

    # concatenate
    full_monthly_aorc_rainfall_list = np.concatenate(full_monthly_aorc_rainfall_list, axis = 0)

    # save as a numpy array
    np.save(save_directory + "/" + "aorc_{0}.npy".format(month), full_monthly_aorc_rainfall_list)
    # flatten to two dimension (time, lon * lat)
    # full_monthly_aorc_rainfall_list = full_monthly_aorc_rainfall_list.reshape(full_monthly_aorc_rainfall_list.shape[0], full_monthly_aorc_rainfall_list.shape[1] * full_monthly_aorc_rainfall_list.shape[2])

    # # save each time series as a separate file
    # for i in range(full_monthly_aorc_rainfall_list.shape[1]):
    #     monthly_aorc_series = full_monthly_aorc_rainfall_list[:, i]
    #     df = pd.DataFrame({"aorc_rainfall":monthly_aorc_series})
    #     df.to_csv(save_directory + "/" + "{0}.csv".format(i), index = False)

