# Generate AORC rainfall fields for each identified rainstorm events.

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing

def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass

def hour_shift(hour, year):
    # To correct the difference in timestamps between ERA5 and AORC
    # if (year == 1979) or (year == 1980):
    #     return hour  # if year is 1979 and 1980, no correction is made

    if hour != 23:
        hour = hour + 1  # if ERA5 time is 00:00, then AORC time is 01:00,
        # if ERA5 time is 23:00, then AORC time is 00:00 next day
    else:
        hour = 0
    return hour

def extract_aorc_rainfall(year):

    directory = r"/home/yliu2232/miss_design_storm/6h_tracking_cesm_res/6h_ar_catalog"
    # load 3-hour ar catalog
    ar_catalog = pd.read_csv(directory + "/" + "{0}_catalog.csv".format(year))
    storm_ids = np.unique(ar_catalog['storm_id'].values)

    print("Start processing aorc rainfall in year {0}".format(year))

    # load aorc lat and lon
    # aorc_flip_lat_df = # pd.read_csv(r"J:\Mississippi_design_storm\processed_data\AORC_AR" + "/" + "aorc_flip_lat.csv")
    # aorc_lon_df = # pd.read_csv(r"J:\Mississippi_design_storm\processed_data\AORC_AR" + "/" + "aorc_lon.csv")
    aorc_flip_lat = np.linspace(50, 29, 630) # aorc_flip_lat_df['aorc_lat'].values
    aorc_lon = np.linspace(-113.16734, -79.068704, 1024) #  aorc_lon_df['aorc_lon'].values

    # storm_index_list = [2]
    for storm_index in range(storm_ids.shape[0]):

        ar_id = storm_ids[storm_index]
        # print("Start extracting aorc rainfall for ar {0}".format(storm_ids[storm_index]))
        storm_record = ar_catalog[ar_catalog['storm_id'] == storm_ids[storm_index]]
        #
        # convert to datetime and get the day
        storm_timesteps = pd.to_datetime(storm_record['time'])
        # get hours
        storm_timestep_hours = storm_timesteps.dt.hour.values
        # get months
        storm_timestep_months = storm_timesteps.dt.month.values

        # if in year 1979, skip storm that starts in January
        if (year == 1979) and (storm_timestep_months[0] == 1):
            # print("AORC data is not available for storm {0}".format(storm_index))
            continue

        if ar_id == 202100135:
            # skip this storm event because missing aorc data at 20211231
            continue

        # get the start hour
        start_hour = storm_timestep_hours[0]
        # get the end end
        end_hour = storm_timestep_hours[-1]

        # correct the start and end hour to match AORC timestamps
        # start_hour = hour_shift(start_hour, year)
        # end_hour = hour_shift(end_hour, year)

        # Extract the date in the format "yyyymmdd"
        storm_day_timesteps = storm_timesteps.dt.strftime('%Y%m%d')
        # get the unique day id
        storm_unique_days = np.unique(storm_day_timesteps)

        full_ar_rainfall_array = []
        full_ar_time_step_list = []

        # extract aorc data by day
        for day_index in range(storm_unique_days.shape[0]):
            aorc_directory = r"/home/group/Datasets/AORCpreciptemp/{0}".format(year)
            # load the day aorc data
            aorc_filename = aorc_directory + "/" + "AORC.{0}.precip.nc".format(storm_unique_days[day_index])

            # if year is not 2021
            if year != 2021:
                # load the day aorc data
                aorc_filename = aorc_directory + "/" + "AORC.{0}.preciptemp.nc".format(storm_unique_days[day_index])
            else:
                aorc_filename = aorc_directory + "/" + "AORC.{0}.precip.nc".format(storm_unique_days[day_index])
            # try to load the data
            try:
                aorc_ds = xr.load_dataset(aorc_filename)
            except:

                return

            # get the data shortname
            if year != 2021:
                aorc_shortname = 'RAINRATE'
            else:
                aorc_shortname = 'precrate'

            # select mississippi range
            aorc_ds_miss = aorc_ds.sel(latitude=slice(29, 50), longitude=slice(-113.17, -79.068704))

            # change the aorc timestamps to be consistent with ERA5
            # shift the aorc by one hour
            # if (year != 1979) and (year != 1980):
            # get the time stamps and shift by -1 hour to align with ERA5 data
            shifted_timestep = aorc_ds_miss['time'].data + np.timedelta64(-1, 'h')
            aorc_ds_miss.update({'time': shifted_timestep})

            # resample it by 3 hours
            resampled_aorc_xarray = aorc_ds_miss.resample(time='6H').sum()
            resampled_aorc_timesteps = resampled_aorc_xarray['time']

            # get the hour steps from 1, 2 to 23, 0
            aorc_hour_timesteps = pd.to_datetime(resampled_aorc_timesteps).hour
            if day_index == 0:  # if it is the first day

                # get the location of starting hour
                start_index = np.argwhere(aorc_hour_timesteps == start_hour)[0][0]

                # check if the total day is one
                if storm_unique_days.shape[0] == 1:
                    # get the location of ending hour
                    end_index = np.argwhere(aorc_hour_timesteps == end_hour)[0][0]
                    # extract the array
                    temp_aorc_array = resampled_aorc_xarray[aorc_shortname].data[start_index:end_index + 1, :, :]

                    # print("Appending {0} staring form {1}:00 to {2}:00".format(storm_unique_days[day_index], start_hour,
                    #                                                            end_hour))

                    # append time steps
                    # full_ar_time_step_list.append(aorc_timesteps[start_index:end_index + 1])

                else:
                    # get the corresponding aorc array
                    temp_aorc_array = resampled_aorc_xarray[aorc_shortname].data[start_index:, :, :]

                    # print("Appending {0} staring form {1}:00 to 21:00".format(storm_unique_days[day_index], start_hour))
                    #
                    # append time steps
                    # full_ar_time_step_list.append(aorc_timesteps[start_index:])

            elif day_index == (storm_unique_days.shape[0] - 1):
                # if it is the last day
                # get the location of ending hour
                end_index = np.argwhere(aorc_hour_timesteps == end_hour)[0][0]
                # extract the array
                temp_aorc_array = resampled_aorc_xarray[aorc_shortname].data[:end_index + 1, :, :]

                # print("Appending {0} staring form 00:00 to {1}:00".format(storm_unique_days[day_index], end_hour))

                # append time steps
                # full_ar_time_step_list.append(aorc_timesteps[:end_index + 1])

            else:
                # get the full aorc array
                temp_aorc_array = resampled_aorc_xarray[aorc_shortname].data
                # print("Appending {0} staring form 00:00 to 21:00".format(storm_unique_days[day_index], end_hour))

                # append time steps
                # full_ar_time_step_list.append(aorc_timesteps)

            # append the temp aorc array to full ar rainfall array
            full_ar_rainfall_array.append(temp_aorc_array)
        # merge
        full_ar_rainfall_array = np.concatenate(full_ar_rainfall_array, axis=0)
        # flip the aorc rainfall array to be consistent with ERA5
        full_ar_rainfall_array = np.flip(full_ar_rainfall_array, axis=1)
        # convert time step to array and concatenate
        # full_ar_time_step_list = np.concatenate(full_ar_time_step_list, axis=0)

        # Create the dataset
        ds = xr.Dataset(
            {'aorc': (['time', 'latitude', 'longitude'], full_ar_rainfall_array)},
            coords={
                'time': storm_timesteps.values,
                'latitude': aorc_flip_lat,
                'longitude': aorc_lon
            },
            attrs={'description': "AORC AR rainfall with storm id {0}, 01:00 means 00:00-01:00 accumulation".format(storm_ids[storm_index])}
        )
        save_directory = r"/home/yliu2232/miss_design_storm/6h_ar_event_cesm_res/{0}/{1}".format(year, ar_id)
        create_folder(save_directory)
        # save the dataset
        ds.to_netcdf(save_directory + '/' + "{0}_aorc.nc".format(storm_ids[storm_index]),
                     encoding={"aorc": {"dtype": "float32", "zlib": True}})


if __name__ == "__main__":
    # year = 2020
    year_list = np.arange(1979, 2022)
    # year_list = np.arange(1979, 2011)

    pool = multiprocessing.Pool(processes = 6)
    pool.map(extract_aorc_rainfall, year_list)

    print("All task is finished.")