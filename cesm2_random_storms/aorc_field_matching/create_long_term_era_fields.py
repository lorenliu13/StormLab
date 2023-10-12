# Create long-term 1979-2021 era covariate and aorc field series for matching
# Yuan Liu
# 2023/07/19


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


month = [12, 1, 2, 3, 4, 5]
year_list = np.arange(1979, 2022)

# Get the field from AORC events
era_ar_mtpr_array_list = []
era_ar_tcwv_array_list = []
era_ar_ivt_array_list = []
era_ar_aorc_array_list = []

for year in year_list:

    print("Process year {0} month {1}".format(year, month))

    # load the catalog
    ar_catalog = pd.read_csv(
        r"/home/yliu2232/miss_design_storm/6h_tracking_cesm_res/6h_ar_catalog" + "/" + "{0}_catalog.csv".format(year))
    ar_ids = np.unique(ar_catalog['storm_id'].values)

    # get ar months
    ar_months = ar_catalog.groupby('storm_id').min()['month'].values
    # get ar ids in specific month
    monthly_ar_ids = ar_ids[np.isin(ar_months, month)]

    # era_ar_number_list.append(monthly_ar_ids.shape[0])

    for ar_id in monthly_ar_ids:

        ar_record = ar_catalog[ar_catalog['storm_id'] == ar_id]
        # convert to datetime and get the day
        storm_timesteps = pd.to_datetime(ar_record['time'])
        # get months
        storm_timestep_months = storm_timesteps.dt.month.values

        if (year == 1979) and (storm_timestep_months[0] == 1):
            print("AORC data is not available for storm {0}".format(ar_id))
            continue

        if ar_id == 202100135:
            continue
        ar_era_xarray_folder = r"/home/yliu2232/miss_design_storm/6h_ar_event_cesm_res/{0}/{1}".format(year, ar_id)

        mtpr_xarray_loc = ar_era_xarray_folder + "/" + "{0}_mean_total_precipitation_rate_cesm_res.nc".format(ar_id)
        tcwv_xarray_loc = ar_era_xarray_folder + "/" + "{0}_total_column_water_vapour_cesm_res.nc".format(ar_id)
        ivt_xarray_loc = ar_era_xarray_folder + "/" + "{0}_vertical_integral_of_water_vapour_flux_cesm_res.nc".format(
            ar_id)

        mtpr_xarray = xr.load_dataset(mtpr_xarray_loc)
        tcwv_xarray = xr.load_dataset(tcwv_xarray_loc)
        ivt_xarray = xr.load_dataset(ivt_xarray_loc)
        # w500_xarray = xr.load_dataset(w500_xarray_loc)

        # aorc_array = aorc_xarray['aorc'].data
        mtpr_array = mtpr_xarray['mtpr'].data * 3600  # unit: mm
        tcwv_array = tcwv_xarray['tcwv'].data  # unit: mm
        ivt_array = ivt_xarray['q'].data

        # load aorc dataset
        aorc_xarray = xr.load_dataset(ar_era_xarray_folder + "/" + "{0}_aorc.nc".format(ar_id))

        aorc_array = aorc_xarray['aorc'].data

        era_ar_mtpr_array_list.append(mtpr_array)
        era_ar_tcwv_array_list.append(tcwv_array)
        era_ar_ivt_array_list.append(ivt_array)
        era_ar_aorc_array_list.append(aorc_array)

era_ar_mtpr_array_list = np.concatenate(era_ar_mtpr_array_list, axis=0)
era_ar_tcwv_array_list = np.concatenate(era_ar_tcwv_array_list, axis=0)
era_ar_ivt_array_list = np.concatenate(era_ar_ivt_array_list, axis=0)
era_ar_aorc_array_list = np.concatenate(era_ar_aorc_array_list, axis=0)


# save the array to npy
save_location = r"/home/yliu2232/miss_design_storm/cesm2_random_storms/nearest_matching/era_1979_2021"
np.save(save_location + "/" + "era_ar_mtpr_array.npy", era_ar_mtpr_array_list)
np.save(save_location + "/" + "era_ar_tcwv_array.npy", era_ar_tcwv_array_list)
np.save(save_location + "/" + "era_ar_ivt_array.npy", era_ar_ivt_array_list)
np.save(save_location + "/" + "era_ar_aorc_array.npy", era_ar_aorc_array_list)


# also create a month list
month = [12, 1, 2, 3, 4, 5]
year_list = np.arange(1979, 2022)

# Get the field from AORC events
era_ar_month_list = []
era_ar_time_step_list = []
ar_id_list = []

for year in year_list:

    print("Process year {0} month {1}".format(year, month))

    # load the catalog
    ar_catalog = pd.read_csv(
        r"/home/yliu2232/miss_design_storm/6h_tracking_cesm_res/6h_ar_catalog" + "/" + "{0}_catalog.csv".format(year))
    ar_ids = np.unique(ar_catalog['storm_id'].values)

    # get ar months
    ar_months = ar_catalog.groupby('storm_id').min()['month'].values
    # get ar ids in specific month
    monthly_ar_ids = ar_ids[np.isin(ar_months, month)]

    # era_ar_number_list.append(monthly_ar_ids.shape[0])

    for ar_id in monthly_ar_ids:

        ar_record = ar_catalog[ar_catalog['storm_id'] == ar_id]
        # convert to datetime and get the day
        storm_timesteps = pd.to_datetime(ar_record['time'])
        # get months
        storm_timestep_months = storm_timesteps.dt.month.values

        if (year == 1979) and (storm_timestep_months[0] == 1):
            print("AORC data is not available for storm {0}".format(ar_id))
            continue

        if ar_id == 202100135:
            continue
        ar_era_xarray_folder = r"/home/yliu2232/miss_design_storm/6h_ar_event_cesm_res/{0}/{1}".format(year, ar_id)

        # append the month list
        # append the time step list
        era_ar_time_step_list.append(storm_timesteps.values)
        era_ar_month_list.append(storm_timestep_months)
        ar_id_list.append([ar_id] * storm_timestep_months.shape[0])

era_ar_time_step_list = np.concatenate(era_ar_time_step_list, axis=0)
era_ar_month_list = np.concatenate(era_ar_month_list, axis=0)
ar_id_list = np.concatenate(ar_id_list, axis=0)
# Create a dataframe to save information
ar_dataframe = pd.DataFrame()
ar_dataframe['ar_id'] = ar_id_list
ar_dataframe['time_step'] = era_ar_time_step_list
ar_dataframe['month'] = era_ar_month_list

# save to csv
ar_dataframe.to_csv(
    r"/home/yliu2232/miss_design_storm/cesm2_random_storms/nearest_matching/era_1979_2021" + "/" + "ar_info_df.csv",
    index=False)

