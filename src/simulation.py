import xarray as xr
import os
import scipy.stats as st
import numpy as np


def rainfall_simulation(noise_array, logic_mu_array, scipy_a_array, scipy_scale_array, scipy_gg_c_array):
    """
    Generate rainfall field
    :param noise_array:
    :param logic_mu_array:
    :param scipy_a_array:
    :param scipy_scale_array:
    :param scipy_gg_c_array:
    :return:
    """

    # convert the gaussian noise field to 0-1 field
    first_prob_array = st.norm.cdf(noise_array, loc=0, scale=1)

    # replace nan with zero
    logic_mu_array = np.where(np.isnan(logic_mu_array), 0, logic_mu_array)
    # compute the probability of dry
    dry_prob_array = 1 - logic_mu_array

    # Compute the second probability
    second_prob_array = first_prob_array - dry_prob_array
    # change all negative probability to np.nan
    second_prob_array = np.where((second_prob_array < 0) | (dry_prob_array == 1), np.nan, second_prob_array)
    # print(second_prob_array[second_prob_array > 1])
    # rescale the remaining to (0, 1)
    second_prob_array = second_prob_array / logic_mu_array

    # add an upper bound for probability
    second_prob_array[second_prob_array > 0.995] = 0.995
    # print(second_prob_array[second_prob_array > 1])

    sim_rainfall_array = np.zeros(noise_array.shape)
    # compute ppf for each time step
    for time_index in range(noise_array.shape[0]):
        # time_index = 0

        curr_second_prob_array = second_prob_array[time_index]
        # get those rainfall probability > 0
        rainfall_mask = ~np.isnan(curr_second_prob_array)

        curr_a_array = scipy_a_array[time_index]
        curr_c_array = scipy_gg_c_array
        curr_scale_array = scipy_scale_array[time_index]

        # get the corresponding GG distribution params
        p = curr_second_prob_array[rainfall_mask]
        # print(p[p > 1])

        # get the rainfall magnitude through inverse pdf
        # an nan appears when p > 1
        rainfall_magnitude_vector = st.gamma.ppf(p, a=curr_a_array[rainfall_mask],
                                                 scale=curr_scale_array[rainfall_mask]
                                                 ) ** (1 / curr_c_array[rainfall_mask])
        rainfall_magnitude_vector[np.isnan(rainfall_magnitude_vector)] = 0
        rainfall_magnitude_vector[np.isinf(rainfall_magnitude_vector)] = 0

        # print(np.sum(np.isnan(rainfall_magnitude_vector)))
        # restore the rainfall to original location
        rainfall_array = np.zeros(curr_second_prob_array.shape)
        rainfall_array[rainfall_mask] = rainfall_magnitude_vector

        sim_rainfall_array[time_index, :, :] = rainfall_array


    return sim_rainfall_array



if __name__ == "__main__":

    # read argument input
    ar_id = 202200018
    realization = 1

    cesm_rainstorm_noise_folder = r"C:\Disk_D\My_drive\Code\My_open_source\StormLab\data\cesm2\cesm2_rainstorm_simulation"
    cesm_rainstorm_distr_param_folder = r"C:\Disk_D\My_drive\Code\My_open_source\StormLab\data\cesm2\cesm2_rainstorms_distr_params"
    # load noise field realization
    noise_xarray = xr.load_dataset(cesm_rainstorm_noise_folder + "/" + "{0}_{1}_noise.nc".format(ar_id, realization))

    # load rainfall probability
    logic_mu_xarray = xr.load_dataset(cesm_rainstorm_distr_param_folder + "/" + "{0}_logit_wet_p.nc".format(ar_id))
    # load scipy parameters
    scipy_a_xarray = xr.load_dataset(cesm_rainstorm_distr_param_folder + "/" + "{0}_scipy_a.nc".format(ar_id))
    scipy_gg_c_xarray = xr.load_dataset(cesm_rainstorm_distr_param_folder + "/" + "{0}_scipy_c.nc".format(ar_id))
    scipy_scale_xarray = xr.load_dataset(cesm_rainstorm_distr_param_folder + "/" + "{0}_scipy_scale.nc".format(ar_id))

    # get the noise array
    noise_array = noise_xarray['aorc'].data
    time_steps = noise_xarray['time'].data
    # get GG and logistic regression model parameters
    logic_mu_array = logic_mu_xarray['aorc'].data
    scipy_a_array = scipy_a_xarray['aorc'].data
    scipy_gg_c_array = scipy_gg_c_xarray['aorc'].data
    scipy_scale_array = scipy_scale_xarray['aorc'].data

    sim_rainfall_array = rainfall_simulation(noise_array, logic_mu_array, scipy_a_array, scipy_scale_array, scipy_gg_c_array)