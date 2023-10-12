# Obtain long-term seasonal series of era5 fields at CESM2 resolution
# Yuan Liu
# 2023/10/11
import xarray as xr
import numpy as np
import xesmf as xe


if __name__ == "__main__":

    # Define the range of years and list of variables to process
    year_list = np.arange(1979, 2022)
    variable_list = ['mean_total_precipitation_rate', 'total_column_water_vapour', 'vertical_integral_of_water_vapour_flux']

    # Define the seasons and their respective months to process
    season_list = {'jja': [6, 7, 8], 'son': [9, 10, 11]}

    # Define the path of era5 files and save locations
    era5_path = r"/home/yliu2232/miss_design_storm/raw_data/ERA5/ERA5_6h"
    save_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/era5_series"

    # Define ERA5 and CESM2 grid resolutions
    era5_lon = np.linspace(-114, -78, 145)
    era5_lat = np.linspace(51, 28, 93)
    cesm_lon = np.linspace(-113.75, -78.75, 29)
    cesm_lat = np.linspace(50.41884817, 28.7434555, 24)


    # Loop through each variable and season
    for variable in variable_list:
        for season in season_list:

            # Array to store processed data for each year
            full_length_era_array = []

            for year in year_list:
                    month_list = season_list[season]

                    # Log current processing status
                    print("Start to process {0} {1}".format(variable, year))
                    # Load ERA5 dataset for the given year and variable
                    era5_xarray = xr.load_dataset(era5_path + "/" + "{0}/ERA5_6H_{1}_{0}.nc".format(year, variable))

                    # Filter data to keep only the desired months
                    subset_era5_xarray = era5_xarray.sel(time=(era5_xarray['time.month'] == month_list[0]) | (era5_xarray['time.month'] == month_list[1]) | (era5_xarray['time.month'] == month_list[2]))

                    # Define target CESM2 grid resolution
                    target_grid = xr.Dataset({
                        'latitude': (('latitude',), cesm_lat),
                        'longitude': (('longitude',), cesm_lon)
                    })

                    # Load pre-defined regridder for conservatively regridding from ERA5 to CESM2 resolution
                    regridder_location = "ERA5_to_CESM2_conservative_regridder.nc"
                    regridder2 = xe.Regridder(subset_era5_xarray, target_grid, 'conservative',
                                              weights=regridder_location)

                    # Regrid the dataset from ERA5 to CESM2 resolution
                    era5_coarse_ds = regridder2(subset_era5_xarray, keep_attrs=True)  # get the regrided dataset

                    # get the variable short name
                    short_name = [k for k in era5_coarse_ds.data_vars.keys()]
                    short_name = short_name[0]
                    # Extract main variable data from the regridded dataset
                    era5_coarse_array = era5_coarse_ds[short_name]

                    # Append the processed data to our array
                    full_length_era_array.append(era5_coarse_array)

            # Combine processed data from all years
            full_length_era_array = np.concatenate(full_length_era_array, axis = 0)
            # Save the combined data
            np.save(save_folder + "/" + "ERA5_1979_2021_{0}_6H_{1}_cesm_res.npy".format(season, variable), full_length_era_array)
