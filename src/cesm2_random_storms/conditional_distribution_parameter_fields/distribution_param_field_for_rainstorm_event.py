# Generate conditional distribution parameter fields based on fitted coefficients and CESM2 large-scale atmospheric variable fields.

import os
import numpy as np
import sys
import xarray as xr
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


def replace_with_nan(array):
    # replace -9999 into nan
    array = np.where(array == -9999, np.nan, array)
    return array


def link(eta, alpha = 1, beta = 0):
    # return distribution parameters from eta
    mu = np.log(1 + np.exp(eta))
    return mu


def fill_nan(arr):
    # perform row wise filling
    nan_idx = np.isnan(arr)  # get a matrix of T/F for nan
    idx = np.where(~nan_idx)[0]  # get the row index of non-nan elements
    arr[nan_idx] = np.interp(np.where(nan_idx)[0], idx, arr[~nan_idx])
    # interpolation:
    # x: np.where(nan_index)[0] # the row index of nan elements
    # xp: idx, the row index of non-nan elements
    # arr[~nan_inx]: the non-nan elements
    return arr


def fill_marginal_nan(array):
    # create a new array
    filled_array = np.zeros(array.shape)
    # perform column wise filling first
    for i in range(array.shape[1]):
        y = array[:, i]
        if np.all(np.isnan(y)):  # if this column is all nan
            filled_array[:, i] = y  # do not change this column
        else:
            filled_array[:, i] = fill_nan(y)  # fill with the nearest value in the column
    # perform rowwise filling
    if np.sum(np.isnan(filled_array)) > 0:  # if there still has nans
        for j in range(array.shape[0]):  # for each row
            x = filled_array[j, :]
            filled_array[j, :] = fill_nan(x)  # fill with the nearest value in the row

    return filled_array


def expand_cesm_grid_to_aorc(mtpr_array, cesm_grid_reference_array):
    """
    Function to expand CESM grid to aorc grid
    :param mtpr_array:
    :param cesm_grid_reference_array:
    :return:
    """
    # expand raw mtpr array to rowwise
    mtpr_array_rowwise = mtpr_array.reshape(-1, 24 * 29)  # dim (1256, 13005)

    # expand the raw mtpr array using the remapping index
    expanded_mtpr_array_rowwise = mtpr_array_rowwise[:, cesm_grid_reference_array]  # new dim (1256, 630 * 1024)

    # reshape it
    expanded_mtpr_array = expanded_mtpr_array_rowwise.reshape(-1, 630, 1024)

    return expanded_mtpr_array



if __name__ == "__main__":

    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the desired save folder relative to the script's directory
    save_folder = os.path.join(script_dir, '../../../output/cesm2_rainstorms_distr_params')
    create_folder(save_folder)

    # Navigate to the desired load folder relative to the script's directory
    distr_param_coeff_folder = os.path.join(script_dir, '../../../data/era5/parameter_coeff_fields')
    cesm_rainstorm_folder = os.path.join(script_dir, '../../../data/cesm2/cesm2_rainstorm_covariates')


    # read argument input
    ar_id = 202200018

    # get coefficients
    # if the coefficient is -9999, that means no data available here, so replace it with np.nan
    alpha_1_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "alpha_1_array.npy"))
    alpha_2_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "alpha_2_array.npy"))
    # alpha_3_array = replace_with_nan(np.load(save_folder + "/" + "alpha_3_array.npy")) sigma coefficient is not used
    alpha_3_array = np.zeros((630, 1024))  # currently there is no alpha 3 array
    alpha_4_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "alpha_4_array.npy"))
    alpha_5_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "alpha_5_array.npy"))

    mu_clim_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "mu_clim_array.npy"))
    sigma_clim_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "sigma_clim_array.npy"))
    gg_c_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "gg_c_array.npy"))

    # load logistic regression coefficients
    logit_intercept_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "logit_intercept_array.npy"))
    logit_mtpr_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "logit_mtpr_array.npy"))
    logit_tcwv_array = replace_with_nan(np.load(distr_param_coeff_folder + "/" + "logit_tcwv_array.npy"))

    # set up AORC coordinates
    aorc_lat = np.linspace(50, 29, 630)
    aorc_lon = np.linspace(-113.16734, -79.068704, 1024)

    # era_lat = np.linspace(51, 28, 93)
    # era_lon = np.linspace(-114, -78, 145)

    cesm_lat = np.linspace(50.41884817, 28.7434555, 24)
    cesm_lon = np.linspace(-113.75, -78.75, 29)

    print("AR id: {0}".format(ar_id))

    # load the aorc and era5 covariate xarray data for the ar event
    mtpr_xarray = xr.load_dataset(cesm_rainstorm_folder + "/" + "{0}_prect_cesm_res.nc".format(ar_id))
    tcwv_xarray = xr.load_dataset(cesm_rainstorm_folder + "/" + "{0}_tmq_cesm_res.nc".format(ar_id))

    # get time steps
    ar_time_stamps = mtpr_xarray['time'].data
    mtpr_array = mtpr_xarray['prect'].data
    tcwv_array = tcwv_xarray['tmq'].data

    # initialize high-res array
    high_res_mtpr_array = np.zeros((ar_time_stamps.shape[0], 630, 1024))
    high_res_tcwv_array = np.zeros((ar_time_stamps.shape[0], 630, 1024))

    for i in range(mtpr_array.shape[0]):
        # interpolate
        high_res_mtpr_array[i] = linear_interpolation(mtpr_array[i], cesm_lon, cesm_lat, aorc_lon, aorc_lat)
        high_res_tcwv_array[i] = linear_interpolation(tcwv_array[i], cesm_lon, cesm_lat, aorc_lon, aorc_lat)

    # create empty array to save data
    full_scipy_a_array = []
    full_scipy_scale_array = []
    full_wet_p_array = []  # compute the probability of dry

    for time_index in range(high_res_mtpr_array.shape[0]):
        # get current mtpr array
        curr_mtpr_array = high_res_mtpr_array[time_index]
        curr_tcwv_array = high_res_tcwv_array[time_index]

        # compute the mu array
        logarg = alpha_2_array + curr_mtpr_array * alpha_4_array + curr_tcwv_array * alpha_5_array
        # variant: let's not use curr_tcwv
        # logarg = alpha_2_array + curr_mtpr_array * alpha_4_array
        mu = mu_clim_array / alpha_1_array * np.log1p(np.expm1(alpha_1_array) * logarg)

        sigma = sigma_clim_array

        # compute current parameter a
        a = mu ** 2 / sigma ** 2
        # compute current parameter scale
        scale = sigma ** 2 / mu

        # compute the wet probability
        eta_array = logit_intercept_array + logit_mtpr_array * curr_mtpr_array + logit_tcwv_array * curr_tcwv_array
        wet_p_array = 1 / (1 + np.exp((-1.0) * eta_array))

        full_scipy_a_array.append(a)
        full_scipy_scale_array.append(scale)
        full_wet_p_array.append(wet_p_array)

    full_scipy_a_array = np.array(full_scipy_a_array)
    full_scipy_scale_array = np.array(full_scipy_scale_array)

    # Create the dataset for scipy a parameter
    scipy_a_ds = xr.Dataset(
        {'aorc': (['time', 'latitude', 'longitude'], full_scipy_a_array)},
        coords={
            'time': ar_time_stamps,
            'latitude': aorc_lat,
            'longitude': aorc_lon
        },
        attrs={'description': "Gamma Scipy a parameter for AORC AR id {0}".format(
            ar_id)}
    )
    # save the dataset
    scipy_a_ds.to_netcdf(save_folder + "/" + "{0}_scipy_a.nc".format(ar_id),
                         encoding={"aorc": {"dtype": "float32", "zlib": True}})

    # Create the dataset for scipy scale parameter
    scipy_scale_ds = xr.Dataset(
        {'aorc': (['time', 'latitude', 'longitude'], full_scipy_scale_array)},
        coords={
            'time': ar_time_stamps,
            'latitude': aorc_lat,
            'longitude': aorc_lon
        },
        attrs={'description': "Gamma Scipy scale parameter for AORC AR id {0}".format(
            ar_id)}
    )
    # save the dataset
    scipy_scale_ds.to_netcdf(save_folder + "/" + "{0}_scipy_scale.nc".format(ar_id),
                             encoding={"aorc": {"dtype": "float32", "zlib": True}})

    # Create the dataset for scipy c parameter
    scipy_c_ds = xr.Dataset(
        {'aorc': (['latitude', 'longitude'], gg_c_array)},
        coords={
            # 'time': ar_time_stamps,
            'latitude': aorc_lat,
            'longitude': aorc_lon
        },
        attrs={'description': "Generalized Gamma Scipy c parameter for AORC AR id {0}".format(
            ar_id)}
    )
    # save the dataset
    scipy_c_ds.to_netcdf(save_folder + "/" + "{0}_scipy_c.nc".format(ar_id),
                         encoding={"aorc": {"dtype": "float32", "zlib": True}})

    # create dataset for dry probability
    logit_wet_p_ds = xr.Dataset(
        {'aorc': (['time', 'latitude', 'longitude'], full_wet_p_array)},
        coords={
            'time': ar_time_stamps,
            'latitude': aorc_lat,
            'longitude': aorc_lon
        },
        attrs={'description': "Dry probability from logistic regression for AORC AR id {0}".format(
            ar_id)}
    )
    # save the dataset
    logit_wet_p_ds.to_netcdf(save_folder + "/" + "{0}_logit_wet_p.nc".format(ar_id),
                             encoding={"aorc": {"dtype": "float32", "zlib": True}})