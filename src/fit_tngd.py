import pandas as pd
import scipy.stats as st
import numpy as np
import os
from CSGD_hybrid import fit_regression_v2
from sklearn.linear_model import LogisticRegression


def fit_csgd_hybrid_distribution(training_df, y, x_list, mu_fix, sigma_fix):


    # keep greater than 0.2 rainfall
    training_data = training_df[training_df[y] > 0.2]
    aorc_series = training_data[y].values
    covars = training_data[x_list].values

    # fit a no-covariate gg distribution
    gg_a, gg_c, gg_loc, gg_scale = st.gengamma.fit(aorc_series, method='MM', floc=0)
    # Compute the mean from ga distribution
    mean_ga = st.gamma.mean(a=gg_a, loc=gg_loc, scale=gg_scale ** gg_c)
    std_ga = st.gamma.std(a=gg_a, loc=gg_loc, scale=gg_scale ** gg_c)
    loc_ga = gg_loc

    reduced_aorc_series = aorc_series ** gg_c
    p_clim = np.array([mean_ga, std_ga, loc_ga])

    obs_prcp = reduced_aorc_series  # aorc data
    output = fit_regression_v2(obs_prcp, covars, p_clim,
                               initguess=False,
                               constrain=False, mu_fix=mu_fix, sigma_fix=sigma_fix)

    # check if the fitting success:
    if output.success == True:
        # generate the quantile plot
        par = output.x

        par = np.hstack([par, p_clim])
        par = np.hstack([par, gg_c])
        # reshape par into (1, number of parameters)
        par = par.reshape((1, par.shape[0]))
    else:
        covars_new = training_data[x_list[0]].values
        # reshape into -1, 1
        covars_new = covars_new.reshape(-1, 1)
        # only use mtpr to fit
        output = fit_regression_v2(obs_prcp, covars_new, p_clim,
                                   initguess=False,
                                   constrain=False, mu_fix=mu_fix, sigma_fix=sigma_fix)
        # generate the quantile plot
        par = output.x
        # append another zero coefficient to match the length
        par = np.hstack([par, 0])
        par = np.hstack([par, p_clim])
        par = np.hstack([par, gg_c])
        # reshape par into (1, number of parameters)
        par = par.reshape((1, par.shape[0]))

    # generate column names
    alpha_col_names = list(np.arange(1, 4 + len(x_list)))
    alpha_col_names += ['mu_clim', 'sigma_clim', 'loc_clim', 'gg_c']

    # generate a dataframe
    par_df = pd.DataFrame(par, columns=alpha_col_names)

    return par_df


def fit_tngd(data:pd.DataFrame, y:str, x_list:list, mu_fix=False, sigma_fix=False):
    """

    :param data: A dataframe of training data
    :param y: Column name of y.
    :param x_list: List of column name of x
    :return:
    """

    # Fit the hybrid csgd model
    single_grid_df = fit_csgd_hybrid_distribution(training_df=data, y=y, x_list=x_list, mu_fix=mu_fix,
                                                  sigma_fix=sigma_fix)

    return single_grid_df



if __name__ == "__main__":


    # Navigate to the desired load folder relative to the script's directory
    era_df_folder = r"C:\Disk_D\My_drive\Code\My_open_source\StormLab\data\era5\fitting_dataframe"

    # local testing:
    batch_index = 1
    aorc_df = pd.read_csv(era_df_folder + "/" + "{0}".format(batch_index) + "/" + "{0}_aorc.csv".format(batch_index))
    mtpr_df = pd.read_csv(era_df_folder + "/" + "{0}".format(batch_index) + "/" + "{0}_mtpr.csv".format(batch_index))
    tcwv_df = pd.read_csv(era_df_folder + "/" + "{0}".format(batch_index) + "/" + "{0}_tcwv.csv".format(batch_index))


    # create the training dataframe
    # get the columns as grid index s
    grid_index_list = aorc_df.columns
    grid_index = grid_index_list[0]

    training_df = pd.DataFrame()
    training_df['aorc'] = aorc_df[str(grid_index)].values
    training_df['mean_total_precipitation_rate'] = mtpr_df[str(grid_index)].values
    training_df['total_column_water_vapour'] = tcwv_df[str(grid_index)].values

    # Fit the hybrid csgd model
    single_grid_df = fit_tngd(data=training_df, y='aorc', x_list=['mean_total_precipitation_rate', 'total_column_water_vapour'], mu_fix=False, sigma_fix=False)

    print(single_grid_df)