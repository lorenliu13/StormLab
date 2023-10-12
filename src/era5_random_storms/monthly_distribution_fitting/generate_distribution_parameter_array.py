# Generate the fitted distribution parameter coefficient fields.
# Yuan Liu
# 06/15/2023


import pandas as pd
import numpy as np
import os


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


def save_array(array, target_location, file_name):
    # target_file
    target_file_location = target_location + "/" + file_name
    np.save(target_file_location, array)



if __name__ == "__main__":

    month = 4

    month_list = [1, 2, 3, 4, 5, 12]

    # if generate param with only mtpr as covariate
    # mtpr_only = False

    for month in month_list:
        print("Start to create parameter fields for month {0}".format(month))
        # merge then into a single dataframe
        full_csgd_param_df = pd.DataFrame()

        # fold_id = 0

        for batch_id in range(1000):
            save_folder = r"/home/yliu2232/miss_design_storm/6h_tngd_fitting/grid_wise_fitting/{0}".format(month)

            csgd_param_df = pd.read_csv(
                save_folder + "/" + "csgd_param_{0}.csv".format(batch_id))
            # remove those has -9999
            # clean_csgd_param_df = csgd_param_df[csgd_param_df['1'] != -9999]
            # concat
            full_csgd_param_df = pd.concat([full_csgd_param_df, csgd_param_df], axis=0)

        # get the array of parameters
        alpha_1_array = full_csgd_param_df['1'].values.reshape(630, 1024)
        alpha_2_array = full_csgd_param_df['2'].values.reshape(630, 1024)

        alpha_4_array = full_csgd_param_df['4'].values.reshape(630, 1024)

        # if mtpr_only == True:
        #     alpha_5_array = np.zeros((630, 1024))
        # else:
        alpha_5_array = full_csgd_param_df['5'].values.reshape(630, 1024)

        gg_c_array = full_csgd_param_df['gg_c'].values.reshape(630, 1024)

        logit_intercept_array = full_csgd_param_df['logit_intercept'].values.reshape(630, 1024)
        logit_mtpr_array = full_csgd_param_df['logit_mtpr'].values.reshape(630, 1024)
        logit_tcwv_array = full_csgd_param_df['logit_tcwv'].values.reshape(630, 1024)

        mu_clim_array = full_csgd_param_df['mu_clim'].values.reshape(630, 1024)
        sigma_clim_array = full_csgd_param_df['sigma_clim'].values.reshape(630, 1024)

        # save the parameter fields
        save_folder = r"/home/yliu2232/miss_design_storm/6h_tngd_fitting/grid_wise_fitting_params/{0}".format(month) + "/" + "full"
        create_folder(save_folder)

        # save the parameters for mean
        save_array(alpha_1_array, save_folder, 'alpha_1_array.npy')
        save_array(alpha_2_array, save_folder, 'alpha_2_array.npy')
        save_array(alpha_4_array, save_folder, 'alpha_4_array.npy')
        save_array(alpha_5_array, save_folder, 'alpha_5_array.npy')
        # save the shape parameter
        save_array(gg_c_array, save_folder, 'gg_c_array.npy')

        # save logistic regression parameters
        save_array(logit_intercept_array, save_folder, 'logit_intercept_array.npy')
        save_array(logit_mtpr_array, save_folder, 'logit_mtpr_array.npy')
        save_array(logit_tcwv_array, save_folder, 'logit_tcwv_array.npy')

        # save the climatological parameters
        save_array(mu_clim_array, save_folder, 'mu_clim_array.npy')
        save_array(sigma_clim_array, save_folder, 'sigma_clim_array.npy')


