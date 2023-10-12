# Obtain long-term seasonal series of cesm2 atmospheric variables
# Yuan Liu
# 2023/10/11
import xarray as xr
import numpy as np
import os
import pandas as pd


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


if __name__ == "__main__":

    # Define the ensemble year and ensemble IDs to process
    ensemble_year = 1251
    ensemble_ids = [16, 17, 18, 19, 20]

    # Define cesm2 data location and save folder
    cesm_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour"
    save_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction"

    # Loop through each ensemble ID
    for ensemble_id in ensemble_ids:

        # Define the list of variables to process
        variable_list = ['prect', 'tmq', 'ivt']
        # Define the seasons and their corresponding months
        season_list = {'djf': [12, 1, 2], 'mam': [3, 4, 5], 'jja': [6, 7, 8], 'son': [9, 10, 11]}

        # Define the periods and their corresponding years
        period_list = {'early':np.arange(1950, 1979), 'current': np.arange(1979, 2022), 'late': np.arange(2022, 2051)}

        # Define the latitude and longitude for ERA5 and CESM datasets
        era5_lon = np.linspace(-114, -78, 145)
        era5_lat = np.linspace(51, 28, 93)
        cesm_lon = np.linspace(-113.75, -78.75, 29)
        cesm_lat = np.linspace(50.41884817, 28.7434555, 24)

        # Loop through each variable, season, and period
        for variable in variable_list:
            for season in season_list:
                for period in period_list:

                    # Extract the year list for the current period
                    year_list = period_list[period]

                    # Initialize arrays to store data
                    full_length_era_array = []

                    full_year_record = []
                    full_month_record = []

                    # Loop through each year in the year list
                    for year in year_list:

                        # Get the months corresponding to the current season
                        month_list = season_list[season]

                        print("Start to process {0} {1}".format(variable, year))
                        # Load the CESM dataset for the current variable and year
                        cesm_xarray = xr.load_dataset(
                            cesm_folder + "/" + "{0}_{1}/{2}/CESM2_{3}_{2}.nc".format(ensemble_year, ensemble_id,
                                    year, variable))

                        # Filter the dataset to include only the months of the current season
                        subset_era5_xarray = cesm_xarray.sel(
                            time=(cesm_xarray['time.month'] == month_list[0]) | (cesm_xarray['time.month'] == month_list[1]) | (
                                        cesm_xarray['time.month'] == month_list[2]))

                        # Extract year and month information from the dataset
                        year_steps = subset_era5_xarray['time.year'].data
                        month_steps = subset_era5_xarray['time.month'].data

                        # Extract the name of the variable from the dataset
                        short_name = [k for k in subset_era5_xarray.data_vars.keys()]
                        short_name = short_name[0]

                        # Extract the data for the variable from the dataset
                        cesm_array = subset_era5_xarray[short_name]

                        # Append the extracted data to the initialized arrays
                        full_length_era_array.append(cesm_array)
                        full_year_record.append(year_steps)
                        full_month_record.append(month_steps)

                    # Combine the data from each year into a single array
                    full_year_record = np.concatenate(full_year_record, axis=0)
                    full_month_record = np.concatenate(full_month_record, axis=0)

                    # Define the save location for the processed data
                    full_length_era_array = np.concatenate(full_length_era_array, axis=0)
                    sub_folder = save_folder + "/" + "{0}_{1}".format(ensemble_year, ensemble_id)

                    # Create the save directory, if it doesn't exist
                    create_folder(sub_folder)
                    # Save the processed data as a .npy file
                    np.save(sub_folder + "/" + "{0}_{1}_{2}_{3}_6H_{4}_cesm_res.npy".format(ensemble_year, ensemble_id, period, season, variable),
                            full_length_era_array)

                    # Create a DataFrame to store year and month information
                    year_month_df = pd.DataFrame()
                    year_month_df['year'] = full_year_record
                    year_month_df['month'] = full_month_record
                    # Save the year and month information as a .csv file
                    year_month_df.to_csv(sub_folder + "/" + "{0}_{1}_{2}_{3}_6H_{4}_year_month_reference.csv".format(
                        ensemble_year, ensemble_id, period, season, variable), index=False)

