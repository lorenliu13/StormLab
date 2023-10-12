# Combine the AORC rainfall fields of all rainstorm events.
# Yuan Liu
# 2023/02/08


import xarray as xr
import pandas as pd
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

            # load ar nc file
            aorc_xarray_loc = r"/home/yliu2232/miss_design_storm/6h_ar_event_cesm_res/{0}/{1}".format(year, ar_id)
            aorc_xarray = xr.load_dataset(aorc_xarray_loc + "/" + "{0}_aorc.nc".format(ar_id))
            aorc_array = aorc_xarray['aorc'].data

            full_monthly_aorc_rainfall_list.append(aorc_array)

    # concatenate
    full_monthly_aorc_rainfall_list = np.concatenate(full_monthly_aorc_rainfall_list, axis = 0)

    # save as a numpy array
    np.save(save_directory + "/" + "aorc_{0}.npy".format(month), full_monthly_aorc_rainfall_list)
