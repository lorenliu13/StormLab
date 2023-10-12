# Generate noise field of AR event using new code
# Yuan Liu
# 2023/05/08

import xarray as xr
import numpy as np
from pyproj import Transformer
from fftgenerators import initialize_nonparam_2d_ssft_filter
from fftgenerators import generate_noise_2d_ssft_filter_new
import pandas as pd
import random
import os
from scipy import interpolate


def linear_interpolation(array, old_x, old_y, new_x, new_y):
    """
    Linear interpolation of current 2-d field. Note that the latitude need to be flipped.
    :param array:
    :param old_x:
    :param old_y:
    :param new_x:
    :param new_y:
    :return:
    """
    # get current array
    curr_flip_array = np.flip(array, axis=0)

    # create interpolation function
    f = interpolate.interp2d(old_x, np.flip(old_y), curr_flip_array, kind = 'linear')

    # use it to interpolate to new grid
    new_z = f(new_x, np.flip(new_y))

    # flip it
    new_z = np.flip(new_z, axis = 0)

    return new_z

def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


def coordinate_transform(lon_data, lat_data):
    """
    Transform geographic coordinate (degree) to  projected coordinate (m)
    :param lon_data: Longitude coordinate of the grid (degree)
    :param lat_data: Latitude coordinate of the grid (degree)
    :return:
    """
    lon_array, lat_array = np.meshgrid(lon_data, lat_data)
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
    lat_prj_array = np.reshape(np.array(lat_prj_array), lat_array.shape)  # projected location in m
    lon_prj_array = np.reshape(np.array(lon_prj_array), lat_array.shape)  # projected location in m

    return lat_prj_array, lon_prj_array


# Compute the lag-1 correlation of the rainfall field
def lag1_autocorr(matrix):
    # Subtract the mean along the first axis
    demeaned_matrix = matrix - matrix.mean(axis=0, keepdims=True)

    # Compute the product of the matrix with its lagged version (shifted along the first axis)
    lagged_product = demeaned_matrix[:-1] * demeaned_matrix[1:]

    # Compute the sums needed for the auto-correlation coefficients
    numerator = np.sum(lagged_product, axis=0)
    denominator = np.sqrt(np.sum(demeaned_matrix[:-1] ** 2, axis=0) * np.sum(demeaned_matrix[1:] ** 2, axis=0))

    # set where denominator = 0 to 1
    numerator[denominator == 0] = 1
    denominator[denominator == 0] = 1
    # Compute the auto-correlation coefficients, handling the case when the denominator is 0
    autocorr_coeffs = numerator / denominator

    return autocorr_coeffs


def noise_generation(prcp_array, raw_cesm_prcp_array, u_array, v_array, lon_data, lat_data, window_size, window_function,
                     overlap_ratio, ssft_war_thr, seed):
    """

    :param prcp_array: precipitation field data, unit: mm/h
    :param raw_cesm_prcp_array: raw precipitation from CESM2, unit: mm/h
    :param u_array: u wind field data, unit: m/h
    :param v_array: v wind field data, unit: m/h
    :param lon_data: longitude coordinate of the data (degree)
    :param lat_data: latittude coordinate of the data (degree)
    :param window_size: size of the local fft window
    :param window_function: the tapering function for local fft window
    :param overlap_ratio: overlap ratio of each local fft window
    :param ssft_war_thr: threshold of wet area ratio to perform local fft
    :param seed: random seed
    :return:
    """
    # create a dataframe to save processing steps
    step_report = pd.DataFrame()
    operation_list = [] # save the fft operation at each step
    war_list = [] # save the global wet area ratio at each step

    # get projected coordinate
    lat_prj_array, lon_prj_array = coordinate_transform(lon_data, lat_data)
    # compute the distance between grids x[i+1] - x[i] (unit: m)
    x_diff = np.diff(lon_prj_array, axis=1)
    # append a column at the begining column
    x_diff = np.insert(x_diff, 0, x_diff[:, 0], axis=1)
    # compute the distance between grids y[i] - y[i+1] (unit: m), note that y[i+1] < y[i], so there is a negative term
    y_diff = -np.diff(lat_prj_array, axis=0)
    # append a row at the begining
    y_diff = np.insert(y_diff, 0, y_diff[0, :], axis=0)

    # get precipitation field size
    x_size = lon_data.shape[0]
    y_size = lat_data.shape[0]

    # get x and y index array
    yind = np.repeat(np.arange(0, y_size), x_size).reshape(y_size, x_size).astype('int16')  # get an array of y index
    xind = np.tile(np.arange(0, x_size), y_size).reshape(y_size, x_size).astype('int16')  # get an array of x index

    # compute the temporal auto-correlation of the rainfall fields (using the raw precipitation array at CESM2 resolution)
    # flat the precipitation array along x and y axis, [time, y, x] into [time, y * x]
    flat_prcp_vector = np.reshape(raw_cesm_prcp_array, (raw_cesm_prcp_array.shape[0], -1))
    # compute pearson correlation
    alpha_list = []
    alpha_list.append(0.5)  # append an initial value of 0.5, this value is not used, just for adding one step here
    for i in range(raw_cesm_prcp_array.shape[0] - 1):
        # get rid of purely dry girds, i.e., those grids have no rain at both time step
        vector_mask = (flat_prcp_vector[i] != 0) | (flat_prcp_vector[i + 1] != 0)
        prev_rain_vector = flat_prcp_vector[i][vector_mask]
        curr_rain_vector = flat_prcp_vector[i + 1][vector_mask]

        corr = np.corrcoef(prev_rain_vector, curr_rain_vector)  # compute pearson correlation coefficient [2,2] matrix
        alpha_list.append(corr[0, 1])  # get the coefficient, which is located at [0, 1]
    alpha_list = np.array(alpha_list)

    # generate random noise field
    # initialize a time series of noise field
    # raw_prcp_noise_array = np.zeros(prcp_array.shape)
    # initialize a time series of zero field for final noise
    final_prcp_noise_array = np.zeros(prcp_array.shape)

    # generate random Gaussian noise with the same dimension as the rainfall array
    randstate = np.random
    randstate.seed(seed)
    raw_noise_field = randstate.randn(prcp_array.shape[0], prcp_array.shape[1], prcp_array.shape[2])

    for time_step in range(prcp_array.shape[0]):

        if time_step == 0:
            # perform local fft
            Fp = initialize_nonparam_2d_ssft_filter(prcp_array[time_step], win_size=window_size,
                                                    win_fun=window_function,
                                                    overlap=overlap_ratio, war_thr=ssft_war_thr)
            # generate correlated random noise
            final_prcp_noise_array[time_step] = generate_noise_2d_ssft_filter_new(Fp, raw_noise_field[time_step],
                                                                                  overlap=overlap_ratio,
                                                                                  win_fun=window_function, )
        else: # if time step is greater than 1, compute advection
            # perform local fft
            Fp = initialize_nonparam_2d_ssft_filter(prcp_array[time_step], win_size=window_size,
                                                    win_fun=window_function,
                                                    overlap=overlap_ratio, war_thr=ssft_war_thr)

            dis_x_grid = np.round(u_array[time_step] / x_diff)  # compute grid displacement of x
            dis_y_grid = np.round(v_array[time_step] / y_diff)  # compute grid displacement of y

            x_advected_index = np.array((xind - dis_x_grid), dtype='int')  # the advected index of x
            x_out_boundary_mask = (x_advected_index < 0) | (
                    x_advected_index > (x_size - 1))  # True if the element is currently empty at current step
            # get the correct index
            x_advected_index_scaled = x_advected_index % x_size

            y_advected_index = np.array((yind - dis_y_grid), dtype='int')
            y_out_boundary_mask = (y_advected_index < 0) | (
                    y_advected_index > (y_size - 1))  # True if the element is empty at current step
            y_advected_index_scaled = y_advected_index % y_size

            # get the noise field at t-1
            previous_noise_field = final_prcp_noise_array[time_step - 1]
            # get the first part of the current noise field at time t
            advected_noise = previous_noise_field[y_advected_index_scaled, x_advected_index_scaled]  # advection
            # get current noise field
            current_noise_field = generate_noise_2d_ssft_filter_new(Fp, raw_noise_field[time_step],
                                                                                  overlap=overlap_ratio,
                                                                                  win_fun=window_function, )
            # fill the empty element with current correlated noise field
            advected_noise = np.where((x_out_boundary_mask) | (y_out_boundary_mask), current_noise_field, advected_noise)

            # compute the final noise
            final_prcp_noise_array[time_step] = alpha_list[time_step] * advected_noise + np.sqrt(
                1 - alpha_list[time_step] ** 2) * current_noise_field

    step_report['alpha'] = np.array(alpha_list)

    return final_prcp_noise_array, step_report


if __name__ == "__main__":
    ensemble_year = 1251
    ensemble_id = 11

    year = 2020
    month = 12
    realization = 1
    ar_id = 202000306 # fixed ar id

    # use ar id to generate a series of random numbers
    # generate a seed field
    random.seed(ar_id)
    # generate a list of random integers using ar_id as seed
    seed_list = []
    for i in range(1000):
        seed_list.append(random.randint(1, 2 ** 20))
    # get the seed for this realization
    seed = seed_list[realization]

    print("Process ensemble_year {0} ensemble_id {1} year {2} month {3}".format(ensemble_year, ensemble_id, year, month))
    print("AR id: {0}".format(ar_id))
    print("Random state: {0}".format(realization))

    # load precipitation data
    raw_cesm_prcp_xarray = xr.load_dataset(
        r"/home/yliu2232/miss_design_storm/cesm2_random_storms/cesm2_ar_covariate_field/{0}_{1}/{2}/{3}".format(ensemble_year, ensemble_id, year, ar_id) + "/" + "{0}_prect_cesm_res.nc".format(ar_id))
    raw_cesm_prcp_array = raw_cesm_prcp_xarray['prect'].data

    # load wind data
    u_xarray = xr.load_dataset(
        r"/home/yliu2232/miss_design_storm/cesm2_random_storms/cesm2_ar_covariate_field/{0}_{1}/{2}/{3}".format(ensemble_year, ensemble_id, year, ar_id) + "/" + "{0}_u850_aorc_res.nc".format(ar_id))
    v_xarray = xr.load_dataset(
        r"/home/yliu2232/miss_design_storm/cesm2_random_storms/cesm2_ar_covariate_field/{0}_{1}/{2}/{3}".format(ensemble_year, ensemble_id, year, ar_id) + "/" + "{0}_v850_aorc_res.nc".format(ar_id))

    # load sr rainfall data
    aorc_array = np.load(r"/home/yliu2232/miss_design_storm/super_resolution/model_prediction/CESM2/noise_generation_sr_rainfall/{0}_{1}/{2}".format(ensemble_year, ensemble_id, year) + "/" + "{0}_sr_rainfall.npy".format(ar_id))

    # get numpy array
    time_steps = u_xarray['time'].data
    u_array = u_xarray['u850'].data
    v_array = v_xarray['v850'].data
    # aorc_array = aorc_xarray['aorc'].data

    # set up AORC coordinates
    aorc_lat = np.linspace(50, 29, 630)
    aorc_lon = np.linspace(-113.16734, -79.068704, 1024)
    # replace -29997 with 0
    # aorc_array = np.where(aorc_array < 0.2, 0, aorc_array)

    window_size = (128, 128)
    window_function = 'tukey'
    overlap_ratio = 0.3
    ssft_war_thr = 0.1

    # generate a copy
    sub_ar_prcp_field_copy = np.where(aorc_array < 0.2, 0, aorc_array)

    final_prcp_noise_array, step_report = noise_generation(sub_ar_prcp_field_copy, raw_cesm_prcp_array, u_array * 3600 * 6,
                                                           v_array * 3600 * 6,
                                                           aorc_lon, aorc_lat, window_size, window_function,
                                                           overlap_ratio, ssft_war_thr, seed)

    # save the noise field output
    save_location = r"/home/yliu2232/miss_design_storm/cesm2_random_storms/noise_field/{0}_{1}/{2}/{3}".format(ensemble_year, ensemble_id, year, ar_id)
    create_folder(save_location)

    # create a ds
    # Create the dataset for scipy scale parameter
    noise_ds = xr.Dataset(
        {'aorc': (['time', 'latitude', 'longitude'], final_prcp_noise_array)},
        coords={
            'time': time_steps,
            'latitude': aorc_lat,
            'longitude': aorc_lon
        },
        attrs={'description': "Noise field {0} for AORC AR id {1}".format(realization,
            ar_id)}
    )

    # save the dataset
    noise_ds.to_netcdf(save_location + "/" + "{0}_{1}_noise.nc".format(ar_id, realization), encoding={"aorc": {"dtype": "float32", "zlib": True}})

    # save the ffst report
    # step_report.to_csv(r"/home/yliu2232/miss_design_storm/random_storms/noise_field/{0}".format(year) + "/" + "{0}_report.csv".format(ar_id), index = False)