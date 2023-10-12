# multi-processing version of AR catalog generation for 6-hour
# Yuan Liu
# 05/09/2023


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.interpolate import interp1d
from pyproj import Transformer
# from AR_trajectory_pan import ar_trajectory
from tqdm import tqdm
import multiprocessing


# compute the IVT weighted centroid
def compute_weighted_centroid(curr_ivt_array, lon_array, lat_array):
    # find the sum of the precipitation values belonging to the storm
    sum_ivt = np.sum(curr_ivt_array)
    # and its intensity weighted centroid
    x_avg = np.sum((lon_array * curr_ivt_array)/sum_ivt)
    y_avg = np.sum((lat_array * curr_ivt_array)/sum_ivt)
    return x_avg, y_avg

# compute the area weighted IVT
def compute_weighted_average(var_array, pixel_area):
    curr_area = np.sum(np.where(var_array > 0, pixel_area, 0))
    if curr_area == 0:
        weighted_avg = 0
    else:
        weighted_avg = np.sum(var_array * pixel_area) / curr_area
    return weighted_avg, curr_area


def remove_leading_trailing_zeros(vec):
    # Find the first non-zero element in the vector， 从前往后数
    first_non_zero = next((i for i, x in enumerate(vec) if x != 0), None)
    # if all the elements are zero
    if first_non_zero is None:
        return []

    # find the last non-zero element in the vector, 从后往前数
    last_non_zero = next((i for i, x in enumerate(vec[::-1]) if x != 0), None)

    # return the index

    return first_non_zero, len(vec) - last_non_zero

# variables:
# 1. Duration (hours):
# 2. Area (sqkm)
# 3. Intense area (IVT > 750 kg/m/s, sqkm)
# 4. IVT intensity (area weighted, kg/m/s)
# 5. Length of the trajectory (km)
# 6. Width = Area / Length (km)
# 7. Sum of the turning angles (degree)
# 8. Speed

def ar_catalog_generation(task:dict):

    # extract the year
    year = task['year']

    print("Start creating ar catalog for {0}".format(year))
    # set up the time interval for the data, which is 3 hours
    time_interval = 6
    # load a time step data
    # load ERA5 precipitation data
    prcp_xarray = xr.open_dataset(
        r"/home/yliu2232/miss_design_storm/raw_data/ERA5/ERA5_6h/{0}".format(year) + "/" + "ERA5_6H_mean_total_precipitation_rate_{0}_cesm_res.nc".format(
            year))
    # get the time step
    time_step = prcp_xarray['time'].data
    # load mississippi basin boundary
    # miss_boundary = np.load(r"/home/yliu2232/code/miss_design_storm/ncg_sbasin_boundary/era5" + "/" + "miss_basin_era5.npy")
    miss_boundary = np.load(
        r"/home/yliu2232/code/miss_design_storm/ncg_sbasin_boundary/cesm2" + "/" + "cesm2_miss_boundary.npy")

    track_array = np.load(r"/home/yliu2232/miss_design_storm/6h_tracking_cesm_res" + "/" + "{0}_tracking.npy".format(year))
    attach_prcp = np.load(
        r"/home/yliu2232/miss_design_storm/6h_tracking_cesm_res" + "/" + "{0}_attached_prcp.npy".format(year))
    ivt_xarray = xr.open_dataset(r"/home/yliu2232/miss_design_storm/raw_data/ERA5/ERA5_6h/{0}".format(
        year) + "/" + "ERA5_6H_vertical_integral_of_water_vapour_flux_{0}_cesm_res.nc".format(year))
    tcwv_xarray = xr.open_dataset(r"/home/yliu2232/miss_design_storm/raw_data/ERA5/ERA5_6h/{0}".format(
        year) + "/" + "ERA5_6H_total_column_water_vapour_{0}_cesm_res.nc".format(year))


    prcp_array = prcp_xarray['mtpr'].data * 3600  # convert unit to mm/hour
    ivt_array = ivt_xarray['q'].data  # unit: kg/m/s
    tcwv_array = tcwv_xarray['tcwv'].data # unit: mm

    # for each storm id, compute storm information
    storm_labels = np.unique(track_array)
    storm_labels = storm_labels[storm_labels != 0]  # remove zero
    print("Total storm number: {0}".format(storm_labels.shape[0]))

    # get projected coordinates
    # ivt_lon = np.linspace(-112, -80.25, 128)
    # ivt_lat = np.linspace(50, 29, 85)
    ivt_lon = np.linspace(-113.75, -78.75, 29)
    ivt_lat = np.linspace(50.41884817, 28.7434555, 24)

    lon_array, lat_array = np.meshgrid(ivt_lon, ivt_lat)
    lon_array_flat = lon_array.flatten()
    lat_array_flat = lat_array.flatten()
    points = []
    for i in range(lon_array_flat.shape[0]):
        points.append((lat_array_flat[i], lon_array_flat[i]))
    # transform lat lon to projected coordinate system
    transformer = Transformer.from_crs(4326, 2163)
    lat_prj_array = []
    lon_prj_array = []
    for pt in transformer.itransform(points):
        lat_prj_array.append(pt[1])
        lon_prj_array.append(pt[0])
    lat_prj_array = np.reshape(np.array(lat_prj_array), lat_array.shape)
    lon_prj_array = np.reshape(np.array(lon_prj_array), lat_array.shape)

    # lat_cell_degree = 0.25  # because it is going down
    # lon_cell_degree = 0.25
    lat_cell_degree = 0.94240838  # because it is going down
    lon_cell_degree = 1.25
    # compute the area of each pixel
    pixel_area = np.cos(lat_array * np.pi / 180) * 111 * 111 * lat_cell_degree * lon_cell_degree

    # compute mississippi area
    total_miss_area = np.sum(np.where(miss_boundary == 1, pixel_area, 0))
    # initialize a dataframe to save all the ar event records
    full_data_frame = pd.DataFrame()

    for i in range(storm_labels.shape[0]):
        storm_label = storm_labels[i]

        # get the number of pixels of storm at each time step
        pixels_num = np.sum(np.isin(track_array, storm_label), axis=(1, 2))

        # initial selection
        # if the duration is less than 48 hours, remove it
        # print("Duration is {0}".format(np.sum(pixels_num != 0) * time_interval))
        # if np.sum(pixels_num != 0) * time_interval < 48:
        #
        #     print("AR duration is less than 48 hours, removed.")
        #     continue

        # extract the storm time sequence
        ar_track_array = track_array[pixels_num != 0]
        prcp_track_array = attach_prcp[pixels_num != 0]

        # Must have certain area over mississippi basin for some duration
        prcp_area_array = np.sum(np.where((miss_boundary == 1) & (prcp_track_array == storm_label), 1, 0) * pixel_area,
                                 axis=(1, 2))

        if np.sum(prcp_area_array > (
                total_miss_area * 0.1)) * time_interval < 24:  # about 10% of the Mississippi basin area
            # print("AR precipitation duration over Mississippi is too short")
            continue

        # obtain the ivt and precipitation field of the current AR event
        storm_ivt_array = np.where(ar_track_array == storm_label, ivt_array[pixels_num != 0], 0)
        storm_prcp_array = np.where(prcp_track_array == storm_label, prcp_array[pixels_num != 0], 0)
        storm_tcwv_array = np.where(ar_track_array == storm_label, tcwv_array[pixels_num != 0], 0)
        # obtain the full ivt field at the time
        complete_ivt_array = ivt_array[pixels_num != 0]

        # get AR time step
        ar_time_step = time_step[pixels_num != 0]
        # transform into pd.Datetimeindex
        ar_time_step = pd.DatetimeIndex(ar_time_step)

        # get the month of the first step
        month = ar_time_step.month
        day = ar_time_step.day

        # compute the area of the AR
        ar_area_array = np.sum(np.where(ar_track_array == storm_label, pixel_area, 0), axis=(1, 2))
        # compute the area of the precipitation
        storm_area_array = np.sum(np.where(prcp_track_array == storm_label, pixel_area, 0), axis=(1, 2))

        # initialize a df for the ar event
        ar_event_df = pd.DataFrame()

        for time_index in range(storm_ivt_array.shape[0]):
            # for time_index in [9]:
            # print("time step {0}".format(time_index))
            # get the ar ivt and precipitation field at the current time step
            curr_ivt_array = storm_ivt_array[time_index]
            curr_prcp_array = storm_prcp_array[time_index]
            curr_tcwv_array = storm_tcwv_array[time_index]

            # get the full ivt field at the current time step
            # curr_complete_ivt_array = complete_ivt_array[time_index]

            # compute the current ar and precipitation area
            curr_ar_area = ar_area_array[time_index]
            curr_storm_area = storm_area_array[time_index]


            avg_ivt, _ = compute_weighted_average(curr_ivt_array, pixel_area)
            avg_prcp, _ = compute_weighted_average(curr_prcp_array, pixel_area)
            avg_tcwv, _ = compute_weighted_average(curr_tcwv_array, pixel_area)

            # compute the high intensity area (> 1000 kg/m/s)
            curr_intense_ivt_array = np.where(curr_ivt_array > 750, storm_ivt_array, 0)
            avg_intense_ivt, intense_ivt_area = compute_weighted_average(curr_intense_ivt_array, pixel_area)

            # compute the area over misissippi
            miss_avg_ivt, miss_ivt_area = compute_weighted_average(np.where(miss_boundary == 1, curr_ivt_array, 0),
                                                                   pixel_area)
            miss_avg_prcp, miss_prcp_area = compute_weighted_average(np.where(miss_boundary == 1, curr_prcp_array, 0),
                                                                     pixel_area)
            miss_avg_tcwv, miss_tcwv_area = compute_weighted_average(np.where(miss_boundary == 1, curr_tcwv_array, 0),
                                                                     pixel_area)

            ivt_x_cent, ivt_y_cent = compute_weighted_centroid(curr_ivt_array, lon_array, lat_array)
            prcp_x_cent, prcp_y_cent = compute_weighted_centroid(curr_prcp_array, lon_array, lat_array)

            storm_id = str(year) + str(storm_label).zfill(5)
            # create a dictionary
            record = dict()
            # Time properties
            record['id'] = [storm_label]
            record['storm_id'] = [storm_id]
            record['time'] = ar_time_step[time_index]
            record['month'] = month[time_index]
            record['day'] = day[time_index]
            # record['duration(hour)'] = duration

            if month[time_index] in [12, 1, 2]:
                season_name = "win"
                season_id = 4
            elif month[time_index] in [3, 4, 5]:
                season_name = 'spr'
                season_id = 1
            elif month[time_index] in [6, 7, 8]:
                season_name = 'sum'
                season_id = 2
            else:
                season_name = 'fal'
                season_id = 3

            record['season'] = season_id

            # centroid properties
            record['ivt_centroid_lon'] = ivt_x_cent
            record['ivt_centroid_lat'] = ivt_y_cent
            record['prcp_centroid_lon'] = prcp_x_cent
            record['prcp_centroid_lat'] = prcp_y_cent

            # Area properties
            record['ivt_area(sqkm)'] = curr_ar_area
            record['prcp_area(sqkm)'] = curr_storm_area
            record['intense_ivt_area(sqkm)'] = intense_ivt_area

            # length properties
            # record['ivt_length(km)'] = ar_length
            # record['ivt_width(km)'] = ar_width
            # record['ivt_lw_ratio(km)'] = ar_lw_ratio

            # curvature
            # record['ivt_sum_angle(deg)'] = angle_sum

            # ivt intensity
            record['avg_ivt_intensity(kg/m/s)'] = avg_ivt
            record['avg_ivt_high_intensity(kg/m/s)'] = avg_intense_ivt

            # prcp intensity
            record['avg_prcp_intensity(mm/h)'] = avg_prcp
            record['avg_tcwv(mm)'] = avg_tcwv

            # over mississippi ivt and prcp
            record['miss_avg_ivt(kg/m/s)'] = miss_avg_ivt
            record['miss_avg_ivt_area(sqkm)'] = miss_ivt_area
            record['miss_avg_prcp(mm/h)'] = miss_avg_prcp
            record['miss_avg_prcp_area(sqkm)'] = miss_prcp_area
            record['miss_avg_tcwv(mm)'] = miss_avg_tcwv
            record['miss_avg_tcwv_area(sqkm)'] = miss_tcwv_area

            # create a dataframe
            record_df = pd.DataFrame(record)

            # append to the full dataframe
            ar_event_df = ar_event_df.append(record_df, ignore_index=True)

            # create a trajectory dataframe
            # track_df = pd.DataFrame(
            #     {"x_cent(m)": full_x_centroid_list,
            #     "y_cent(m)": full_y_centroid_list,
            #     "x_cent(deg)": x_centroid_deg_list,
            #     "y_cent(deg)": y_centroid_deg_list}
            # )
            # track_df.to_csv(r"J:\Mississippi_design_storm\ar_catalog\traject" + "/" + "{0}_{1}_track.csv".format(storm_id, time_index), index=False)

        # Get the computed precipitation area over the mississippi basin
        miss_area_list = ar_event_df['miss_avg_prcp_area(sqkm)'].values
        # print(miss_area_list)

        # apply a filter on the current miss prcp area, if the prcp area < 10% total basin area, consider it as 0
        miss_area_list[miss_area_list < (total_miss_area * 0.1)] = 0
        # remove the very first and end zeros in miss prcp area
        first_non_zero_index, last_non_zero_index = remove_leading_trailing_zeros(miss_area_list)
        # adjust the dataframe based on new range
        ar_event_df = ar_event_df.iloc[first_non_zero_index:last_non_zero_index]
        # assign new duration
        ar_event_df['duration(hour)'] = np.ones(ar_event_df.shape[0]) * ar_event_df.shape[0] * time_interval # length of data times time interval

        # check the duration should be greater than 24 hours, skip the event
        if ar_event_df.shape[0] * time_interval < 24:
            continue

        # append it to the full dataframe
        full_data_frame = full_data_frame.append(ar_event_df, ignore_index=True)

    # save to csv
    full_data_frame.to_csv(r"/home/yliu2232/miss_design_storm/6h_tracking_cesm_res/6h_ar_catalog" + "/" + "{0}_catalog.csv".format(year),
                           index=False)



# creat a task, whcih is a dictionary containing year information
def create_task(year:int):
    task = {"year": year}
    return task


def main():
    # The main code to do multiple processing

    year_list = np.arange(1979, 2022)
    task_list = []
    for year in year_list:
        task = create_task(year) # create the task
        task_list.append(task) # append the task into list

    # Use time processes at the same time
    pool = multiprocessing.Pool(processes = 10)
    pool.map(ar_catalog_generation, task_list)

    print("All task is finished.")



if __name__ == "__main__":

    main()













