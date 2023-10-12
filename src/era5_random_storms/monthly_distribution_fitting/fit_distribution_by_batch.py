# Load the dataframe for the current batch (containing 1,000 grids time series of ERA5 variables or AORC rainfall).
# Perform distribution fitting for each grid in the batch.
# Yuan Liu
# 03/28/2023


import pandas as pd
import scipy.stats as st
import numpy as np
import os
from CSGD_hybrid import fit_regression_v2
from sklearn.linear_model import LogisticRegression


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error

        pass


def fit_csgd_hybrid_distribution(training_df, var_name, mu_fix, sigma_fix):

    # training_df = pd.read_csv("{0}.csv".format(grid_index))

    # get the first element
    first_rainfall = training_df['aorc'].values[0]

    if first_rainfall != -59994:
        # keep greater than 0.2 rainfall
        training_data = training_df[training_df['aorc'] > 0.2]
        aorc_series = training_data['aorc'].values
        covars = training_data[var_name].values

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
            covars_new = training_data['mean_total_precipitation_rate'].values
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

    else:
        par = np.ones((1, 3 + len(var_name) + 3 + 1)) * -9999# alpha 1-3, covariate coefficients, 3 stationary parameter, 1 gg_c parameter

    # generate column names
    alpha_col_names = list(np.arange(1, 4 + len(var_name)))
    alpha_col_names += ['mu_clim', 'sigma_clim', 'loc_clim', 'gg_c']

    # generate a dataframe
    par_df = pd.DataFrame(par, columns=alpha_col_names)

    return par_df



def logistic_regression(training_df):
    # get the first element
    first_rainfall = training_df['aorc'].values[0]

    if first_rainfall != -59994:
        # Fit a logistic regression model using sklearn
        y = np.where(training_df['aorc'].values > 0.2, 1, 0)
        X = training_df[['mean_total_precipitation_rate', 'total_column_water_vapour']].values
        clf = LogisticRegression(random_state=0).fit(X, y)
        # Get the logistic regression coefficients
        logit_intercept = clf.intercept_[0]
        mtpr_coef = clf.coef_[0, 0]
        tcwv_coef = clf.coef_[0, 1]
        # tcwv_coef = 0

    else:
        logit_intercept = -9999
        mtpr_coef = -9999
        tcwv_coef = -9999

    return logit_intercept, mtpr_coef, tcwv_coef


if __name__ == "__main__":

    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the desired save folder relative to the script's directory
    save_folder = os.path.join(script_dir, '../../../output/era5_fitted_distribution_params')
    create_folder(save_folder)

    # Navigate to the desired load folder relative to the script's directory
    era_df_folder = os.path.join(script_dir, '../../../data/era5/fitting_dataframe')


    # local testing:
    batch_index = 1
    aorc_df = pd.read_csv(era_df_folder + "/" + "{0}".format(batch_index) + "/" + "{0}_aorc.csv".format(batch_index))
    mtpr_df = pd.read_csv(era_df_folder + "/" + "{0}".format(batch_index) + "/" + "{0}_mtpr.csv".format(batch_index))
    tcwv_df = pd.read_csv(era_df_folder + "/" + "{0}".format(batch_index) + "/" + "{0}_tcwv.csv".format(batch_index))

    # set the variable name for large-scale atmospheric variables
    var_name = ['mean_total_precipitation_rate', 'total_column_water_vapour']
    mu_fix = False # the mean is varying with time
    sigma_fix = True # the standard deviation is fixed

    # get the columns as grid index s
    grid_index_list = aorc_df.columns

    # create a full dataframe
    batch_grid_df = pd.DataFrame()

    ######################### IMPORTANT: this example only runs 10 grids to save time! #################################
    for grid_index in grid_index_list[0:10]:
    # Not run: for grid_index in grid_index_list:

        # create the training dataframe
        training_df = pd.DataFrame()
        training_df['aorc'] = aorc_df[str(grid_index)].values
        training_df['mean_total_precipitation_rate'] = mtpr_df[str(grid_index)].values
        training_df['total_column_water_vapour'] = tcwv_df[str(grid_index)].values

        # Fit the hybrid csgd model
        single_grid_df = fit_csgd_hybrid_distribution(training_df=training_df, var_name=var_name, mu_fix=mu_fix, sigma_fix=sigma_fix)

        # Fit the logistic regression model
        logit_intercept, mtpr_coef, tcwv_coef = logistic_regression(training_df)
        # Save the logistic regression coefficient
        single_grid_df['logit_intercept'] = [logit_intercept]
        single_grid_df['logit_mtpr'] = [mtpr_coef]
        single_grid_df['logit_tcwv'] = [tcwv_coef]

        # Add the grid id
        single_grid_df['grid_id'] = [grid_index]
        single_grid_df['batch_id'] = [batch_index]

        batch_grid_df = batch_grid_df.append(single_grid_df, ignore_index=True)

    # save the fit_param_df
    batch_grid_df.to_csv(save_folder + "/" + "distr_param_batch_{0}.csv".format(batch_index), index=False)