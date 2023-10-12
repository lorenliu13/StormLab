# Split the long-term bias-corrected CESM2 into yearly array saved by netcdf files.
# Yuan Liu
# 2023/10/11
import numpy as np
import xarray as xr
import os
import pandas as pd


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


# Compute the empirical distribution
def ecdf(data):
    n = data.shape[0]
    ecdf_series = np.arange(1, n + 1) / (n + 1)

    return ecdf_series


if __name__ == "__main__":

    ensemble_year = 1251
    # ensemble_id = 13
    ensemble_ids = np.arange(16, 21)

    for ensemble_id in ensemble_ids:
        period_list = {'early':np.arange(1950, 1979), 'current': np.arange(1979, 2022), 'late': np.arange(2022, 2051)}
        variable_list = {'mean_total_precipitation_rate': 'prect', 'total_column_water_vapour': 'tmq',
                         'vertical_integral_of_water_vapour_flux': 'ivt'}
        # season_list = {'djf': [12, 1, 2], 'mam': [3, 4, 5]}
        season_list = {'djf': [12, 1, 2], 'mam': [3, 4, 5], 'jja': [6, 7, 8], 'son': [9, 10, 11]}
        # month_list = np.arange(1, 13)

        for era_variable in variable_list:
            cesm_variable = variable_list[era_variable]
            for period in period_list:

                season_full_array_dict = {}
                season_full_array_year_month_df_dict = {}
                # load for seasons array
                # for season in season_list:
                for season in season_list:
                    # load bias-corrected full annual year series
                    save_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/bias_corrected_series/{0}_{1}".format(
                        ensemble_year, ensemble_id)
                    full_year_season_bs_cesm_array = np.load(
                        save_folder + "/" + "{0}_{1}_{2}_{3}_6H_{4}_cesm_res_bs.npy".format(ensemble_year, ensemble_id,
                                                                                            period, season, cesm_variable))
                    csv_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/{0}_{1}".format(
                        ensemble_year, ensemble_id)
                    # load the dataframe that record the year and month of each time step
                    full_year_season_reference_df = pd.read_csv(csv_folder + "/" + "{0}_{1}_{2}_{3}_6H_{4}_year_month_reference.csv".format(ensemble_year, ensemble_id,
                                                                                            period, season, cesm_variable))
                    season_full_array_year_month_df_dict[season] = full_year_season_reference_df
                    season_full_array_dict[season] = full_year_season_bs_cesm_array


                year_list = period_list[period]
                for year in year_list:
                    print("Start processing {0} {1}".format(cesm_variable, year))
                    # load raw cesm data
                    # load precipitation dataset
                    cesm_xarray = xr.load_dataset(
                        r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour/{0}_{1}/{2}/CESM2_{3}_{2}.nc".format(
                            ensemble_year, ensemble_id,
                            year, cesm_variable))

                    full_year_bs_corrected_cesm_array = []

                    month_list = np.arange(1, 13)
                    for month in month_list:
                        # check the month in which season
                        if month in [12, 1, 2]:
                            season_name = "djf"
                        elif month in [3, 4, 5]:
                            season_name = 'mam'
                        elif month in [6, 7, 8]:
                            season_name = 'jja'
                        else:
                            season_name = 'son'

                        # get the df
                        cuur_full_year_season_reference_df = season_full_array_year_month_df_dict[season_name]
                        # get year series and month series
                        cuur_year_series = cuur_full_year_season_reference_df['year'].values
                        curr_month_series = cuur_full_year_season_reference_df['month'].values

                        # get the corresponding mask
                        curr_mask = (cuur_year_series == year) & (curr_month_series == month)

                        # get current array data
                        curr_month_bs_array = season_full_array_dict[season_name][curr_mask, :, :]

                        # append to the entire year array
                        full_year_bs_corrected_cesm_array.append(curr_month_bs_array)

                    full_year_bs_corrected_cesm_array = np.concatenate(full_year_bs_corrected_cesm_array, axis=0)
                    # replace the data in the raw cesm xarray
                    cesm_xarray[cesm_variable].data = full_year_bs_corrected_cesm_array

                    # save it to folder
                    save_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/bias_corrected_series/{0}_{1}/{2}".format(
                        ensemble_year, ensemble_id, year)
                    create_folder(save_folder)
                    cesm_xarray.to_netcdf(save_folder + "/" + "CESM2_{0}_{1}_bs.nc".format(year, cesm_variable), encoding={cesm_variable: {"dtype": "float32", "zlib": True}})



