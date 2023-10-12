# Bias-correct the long-term series of CESM2 variable array against ERA5 using the CDF-t method.
# Yuan Liu
# 2023/10/11
import numpy as np
import os
from scipy.interpolate import interp1d
import multiprocessing


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


def delta_qm_stationary(cesm_tcwv_series, era_tcwv_series):
    """
    Quantile mapping without adjustment factor
    :param cesm_tcwv_series:
    :param era_tcwv_series:
    :return:

    """
    # compute the ecdf
    sort_cesm_ecdf = ecdf(cesm_tcwv_series)
    unsort_cesm_ecdf = sort_cesm_ecdf[np.argsort(np.argsort(cesm_tcwv_series))]

    # compute the ecdf of era5
    sort_era_ecdf = ecdf(era_tcwv_series)
    unsort_era_ecdf = sort_era_ecdf[np.argsort(np.argsort(era_tcwv_series))]

    # get the maximum era ecdf
    max_era_ecdf = np.max(unsort_era_ecdf)
    min_era_ecdf = np.min(unsort_era_ecdf)

    # Generate the interpolation function
    f = interp1d(unsort_era_ecdf, era_tcwv_series)
    # replace the outside ecdf to maximum or minimum values
    unsort_cesm_ecdf[unsort_cesm_ecdf >= max_era_ecdf] = max_era_ecdf
    unsort_cesm_ecdf[unsort_cesm_ecdf <= min_era_ecdf] = min_era_ecdf
    # get the quantile mapped series
    mapped_series = f(unsort_cesm_ecdf)

    return mapped_series


def cdf_t_nonstationary(cesm_tcwv_series, era_tcwv_series, reference_cesm_tcwv_series):
    """
    CDF Transform quantile mapping for non-stationary data
    :param cesm_tcwv_series: current period cesm series
    :param era_tcwv_series: era series
    :param reference_cesm_tcwv_series:  reference period cesm series
    :return: quantile mapped series
    """

    # compute the ecdf
    sort_cesm_ecdf = ecdf(cesm_tcwv_series)
    unsort_cesm_ecdf = sort_cesm_ecdf[np.argsort(np.argsort(cesm_tcwv_series))]

    # compute the ecdf of era5
    sort_era_ecdf = ecdf(era_tcwv_series)
    unsort_era_ecdf = sort_era_ecdf[np.argsort(np.argsort(era_tcwv_series))]

    # get reference period cesm series
    ref_cesm_ecdf = ecdf(reference_cesm_tcwv_series)
    unsort_ref_cesm_ecdf = ref_cesm_ecdf[np.argsort(np.argsort(reference_cesm_tcwv_series))]

    # interpolate between reference cesm ecdf and cesm values
    cesm_ref_f = interp1d(unsort_ref_cesm_ecdf, reference_cesm_tcwv_series)
    # compute the corresponding value of cesm data in reference period
    inverse_FGh = cesm_ref_f(unsort_cesm_ecdf)

    # fit a relationship in ERA5 (Fsh)
    era_f = interp1d(era_tcwv_series, unsort_era_ecdf, bounds_error=False,
                     fill_value=(np.min(unsort_era_ecdf), np.max(unsort_era_ecdf)))
    # if x goes out of the bound, use the maximum and minimum probability form era5 ECDF

    # Compute the Fsf
    Fsf = era_f(inverse_FGh) # if value exceed the ERA5 range, use the minimum prob and maximum prob in ERA5 series

    # generate interpolate function
    f = interp1d(Fsf, cesm_tcwv_series, bounds_error=False,
                 fill_value=(np.min(cesm_tcwv_series), np.max(cesm_tcwv_series)))

    # compute the mapped series
    mapped_series = f(unsort_cesm_ecdf)

    return mapped_series


def run_task(task):

    ensemble_year = task['ensemble_year']
    ensemble_id = task['ensemble_id']
    quantile_mapping(ensemble_year, ensemble_id)


def quantile_mapping(ensemble_year, ensemble_id):
    period_list = {'early': np.arange(1950, 1979), 'current': np.arange(1979, 2022), 'late': np.arange(2022, 2051)}
    variable_list = {'mean_total_precipitation_rate': 'prect', 'total_column_water_vapour': 'tmq',
                     'vertical_integral_of_water_vapour_flux': 'ivt'}

    season_list = {'djf': [12, 1, 2], 'mam': [3, 4, 5], 'jja': [6, 7, 8], 'son': [9, 10, 11]}
    # month_list = np.arange(1, 13)

    for period in period_list:
        # for month in month_list:
        for season in season_list:
            for era_variable in variable_list:
                cesm_variable = variable_list[era_variable]

                print("Start to processing {0} {1} {2}".format(period, season, cesm_variable))

                # load CESM and ERA5 series
                file_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/{0}_{1}".format(
                    ensemble_year, ensemble_id)
                era_file_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/era5_series"

                # if the period is early or late period
                if (period == 'early') | (period == 'late'):

                    raw_cesm_tcwv_array = np.load(
                        file_folder + "/" + "{0}_{1}_{2}_{3}_6H_{4}_cesm_res.npy".format(ensemble_year, ensemble_id,
                                                                                         period, season, cesm_variable))
                    raw_era_tcwv_array = np.load(
                        era_file_folder + "/" + "ERA5_1979_2021_{0}_6H_{1}_cesm_res.npy".format(season, era_variable))

                    # if ERA5 variable is precipitation
                    if cesm_variable == 'prect':
                        raw_era_tcwv_array = raw_era_tcwv_array * 3600  # convert the unit to mm

                    # Load reference period (1979-2021) cesm array
                    reference_cesm_tcwv_array = np.load(
                        file_folder + "/" + "{0}_{1}_current_{2}_6H_{3}_cesm_res.npy".format(ensemble_year, ensemble_id,
                                                                                             season, cesm_variable))

                    # initialize an zero array
                    remap_cesm_tcwv_array = np.zeros(raw_cesm_tcwv_array.shape)

                    for lat_index in range(remap_cesm_tcwv_array.shape[1]):
                        for lon_index in range(remap_cesm_tcwv_array.shape[2]):
                            # get time series at this pixel
                            era_tcwv_series = raw_era_tcwv_array[:, lat_index, lon_index]
                            cesm_tcwv_series = raw_cesm_tcwv_array[:, lat_index, lon_index]
                            reference_cesm_tcwv_series = reference_cesm_tcwv_array[:, lat_index, lon_index]

                            # save the adjust series
                            remap_cesm_tcwv_array[:, lat_index, lon_index] = cdf_t_nonstationary(cesm_tcwv_series,
                                                                                                 era_tcwv_series,
                                                                                                 reference_cesm_tcwv_series)

                else:

                    # if the period is current
                    raw_cesm_tcwv_array = np.load(
                        file_folder + "/" + "{0}_{1}_{2}_{3}_6H_{4}_cesm_res.npy".format(ensemble_year, ensemble_id,
                                                                                         period, season, cesm_variable))
                    raw_era_tcwv_array = np.load(
                        era_file_folder + "/" + "ERA5_1979_2021_{0}_6H_{1}_cesm_res.npy".format(season, era_variable))

                    # if ERA5 variable is precipitation
                    if cesm_variable == 'prect':
                        raw_era_tcwv_array = raw_era_tcwv_array * 3600  # convert the unit to mm

                    # initialize an zero array
                    remap_cesm_tcwv_array = np.zeros(raw_cesm_tcwv_array.shape)

                    for lat_index in range(remap_cesm_tcwv_array.shape[1]):
                        for lon_index in range(remap_cesm_tcwv_array.shape[2]):
                            # get time series at this pixel
                            era_tcwv_series = raw_era_tcwv_array[:, lat_index, lon_index]
                            cesm_tcwv_series = raw_cesm_tcwv_array[:, lat_index, lon_index]

                            # save the remapped series
                            remap_cesm_tcwv_array[:, lat_index, lon_index] = delta_qm_stationary(cesm_tcwv_series,
                                                                                                 era_tcwv_series)

                # create a folder to save
                save_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/bias_corrected_series/{0}_{1}".format(
                    ensemble_year, ensemble_id)
                create_folder(save_folder)
                np.save(save_folder + "/" + "{0}_{1}_{2}_{3}_6H_{4}_cesm_res_bs.npy".format(ensemble_year, ensemble_id,
                                                                                            period, season,
                                                                                            cesm_variable),
                        remap_cesm_tcwv_array)


if __name__ == "__main__":

    ensemble_year = 1251
    ensemble_ids = np.arange(16, 21)

    task_list = []
    for ensemble_id in ensemble_ids:
        task = {'ensemble_id': ensemble_id, 'ensemble_year': ensemble_year}
        task_list.append(task)

    pool = multiprocessing.Pool(processes=6)
    pool.map(run_task, task_list)
    print("All task is finished.")




