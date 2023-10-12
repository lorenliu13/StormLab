# Sample the AORC fields for each CESM2 rainstorms based on k nearest neighbor method.
# Yuan Liu
# 2023/07/06


import xarray as xr
import numpy as np
import pandas as pd
import os
import multiprocessing


def compute_euclidean_distance(A, B):
    distances = np.array([np.linalg.norm(A - B[i]) for i in range(B.shape[0])])
    return distances


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


def extract_match_aorc_rainfall_index(ensemble_year, ensemble_id):


    # load full length aorc ar-event
    save_location = r"/home/yliu2232/miss_design_storm/cesm2_random_storms/nearest_matching/era_1979_2021"
    era_ar_mtpr_array_list = np.load(save_location + "/" + "era_ar_mtpr_array.npy")
    era_ar_tcwv_array_list = np.load(save_location + "/" + "era_ar_tcwv_array.npy")
    era_ar_ivt_array_list = np.load(save_location + "/" + "era_ar_ivt_array.npy")
    era_ar_aorc_array_list = np.load(save_location + "/" + "era_ar_aorc_array.npy")

    # era_ar_u_array_list = np.load(save_location + "/" + "era_ar_u_array.npy")
    # era_ar_v_array_list = np.load(save_location + "/" + "era_ar_v_array.npy")

    # load era ar months df
    ar_info_df = pd.read_csv(
        r"/home/yliu2232/miss_design_storm/cesm2_random_storms/nearest_matching/era_1979_2021" + "/" + "ar_info_df.csv")
    era_ar_months = ar_info_df['month'].values
    era_ar_time_steps = ar_info_df['time_step'].values


    # year_list = np.arange(1950, 2051)
    month_list = [1, 2, 3, 4, 5, 12]
    year_list = np.arange(1950, 2051)
    # load AR catalogs
    for year in year_list:
        print("Start processing {0} {1} {2}".format(ensemble_year, ensemble_id, year))
        # load the catalog
        ar_catalog = pd.read_csv(
            r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_catalog_bs/{0}_{1}".format(ensemble_year,
                                                                                                   ensemble_id) + "/" + "{0}_catalog.csv".format(
                year))
        ar_ids = np.unique(ar_catalog['storm_id'].values)
        # get ar months
        ar_months = ar_catalog.groupby('storm_id').min()['month'].values
        # get ar ids in specific month
        monthly_ar_ids = ar_ids[np.isin(ar_months, month_list)]


        # create a folder
        rainfall_save_folder = r"/home/yliu2232/miss_design_storm/cesm2_random_storms/nearest_matching/match_aorc_rainfall/{0}_{1}/{2}".format(ensemble_year,ensemble_id,year)
        create_folder(rainfall_save_folder)

        for ar_id in monthly_ar_ids:

            ar_record = ar_catalog[ar_catalog['storm_id'] == ar_id]
            # get ar month
            ar_month = ar_record['month'].values[0]
            if ar_month == 12:
                nearby_months = [12, 1, 11]
            elif ar_month == 1:
                nearby_months = [12, 1, 2]
            elif ar_month == 2:
                nearby_months = [1, 2, 3]
            elif ar_month == 3:
                nearby_months = [2, 3, 4]
            elif ar_month == 4:
                nearby_months = [3, 4, 5]
            elif ar_month == 5:
                nearby_months = [4, 5, 6]


            ar_era_xarray_folder = r"/home/yliu2232/miss_design_storm/cesm2_random_storms/cesm2_ar_covariate_field_bs/{0}_{1}/{2}/{3}".format(ensemble_year, ensemble_id, year, ar_id)
            mtpr_xarray_loc = ar_era_xarray_folder + "/" + "{0}_prect_cesm_res.nc".format(ar_id)
            tcwv_xarray_loc = ar_era_xarray_folder + "/" + "{0}_tmq_cesm_res.nc".format(ar_id)
            ivt_xarray_loc = ar_era_xarray_folder + "/" + "{0}_ivt_cesm_res.nc".format(ar_id)
            # u_xarray_loc = ar_era_xarray_folder + "/" + "{0}_u850_cesm_res.nc".format(ar_id)
            # v_xarray_loc = ar_era_xarray_folder + "/" + "{0}_v850_cesm_res.nc".format(ar_id)

            mtpr_xarray = xr.load_dataset(mtpr_xarray_loc)
            tcwv_xarray = xr.load_dataset(tcwv_xarray_loc)
            ivt_xarray = xr.load_dataset(ivt_xarray_loc)
            # u850_xarray = xr.load_dataset(u_xarray_loc)
            # v850_xarray = xr.load_dataset(v_xarray_loc)

            mtpr_array = mtpr_xarray['prect'].data  # unit: mm
            tcwv_array = tcwv_xarray['tmq'].data  # unit: mm
            ivt_array = ivt_xarray['ivt'].data  # unit: kg/m/s
            # u_array = u850_xarray['u850'].data  # unit: m/s
            # v_array = v850_xarray['v850'].data

            match_aorc_array_list = []

            # get a current precipitation array
            for time_index in range(mtpr_array.shape[0]):
                curr_cesm_mtpr_array = mtpr_array[time_index]
                curr_cesm_tcwv_array = tcwv_array[time_index]
                curr_cesm_ivt_array = ivt_array[time_index]
                # curr_cesm_u_array = u_array[time_index]
                # curr_cesm_v_array = v_array[time_index]

                # Compute the distance
                mtpr_distances = compute_euclidean_distance(curr_cesm_mtpr_array, era_ar_mtpr_array_list)
                tcwv_distances = compute_euclidean_distance(curr_cesm_tcwv_array, era_ar_tcwv_array_list)
                ivt_distances = compute_euclidean_distance(curr_cesm_ivt_array, era_ar_ivt_array_list)
                # u_distances = compute_euclidean_distance(curr_cesm_u_array, era_ar_u_array_list)
                # v_distances = compute_euclidean_distance(curr_cesm_v_array, era_ar_v_array_list)

                # Compute total distance
                total_distance = mtpr_distances + tcwv_distances + ivt_distances # + u_distances + v_distances

                # select where belongs to nearby months
                selected_total_distance = np.where(np.isin(era_ar_months, nearby_months), total_distance, 99999)

                # Get the indices of the 5 closest slices
                indices = np.argsort(selected_total_distance)[:5]

                closest_5_distance = np.sort(selected_total_distance)[:5]
                probabilities = 1 / closest_5_distance / np.sum(1 / closest_5_distance)
                # random sample an element from the vector
                random_sample_index = np.random.choice(indices, p=probabilities)

                # Get the era value at corresponding index
                match_aorc_array = era_ar_aorc_array_list[random_sample_index]
                match_aorc_array_list.append(match_aorc_array)

            match_aorc_array_list = np.array(match_aorc_array_list)

            # save the aorc rainfall
            np.save(
                rainfall_save_folder + "/" + "{0}_sr_rainfall.npy".format(
                    ar_id), match_aorc_array_list)


def run_task(task):

    ensemble_year = task['ensemble_year']
    ensemble_id = task['ensemble_id']
    # year = task['year']

    extract_match_aorc_rainfall_index(ensemble_year, ensemble_id)


if __name__ == "__main__":

    ensemble_year = 1251
    # ensemble_id = 11
    ensemble_ids = np.arange(16, 21)

    task_list = []
    for ensemble_id in ensemble_ids:

        task = {'ensemble_id':ensemble_id, 'ensemble_year':ensemble_year}
        task_list.append(task)

    pool = multiprocessing.Pool(processes=5)
    pool.map(run_task, task_list)
    print("All task is finished.")