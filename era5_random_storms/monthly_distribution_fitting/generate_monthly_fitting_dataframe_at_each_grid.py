# Generate monthly dataframe for distribution fitting at each grid cell
# Yuan Liu
# 2023/05/11


import numpy as np
import os
import pandas as pd
import multiprocessing


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


def generate_training_df(task):

    month = task['month']
    variable = task['variable']
    variable_short_name = task['short_name']

    print("Processing month {0} variable {1}".format(month, variable_short_name))
    # load data array
    # set the file location
    file_location = r"/home/yliu2232/miss_design_storm/6h_monthly_series/{0}".format(month)

    if variable_short_name == 'aorc':
        var_array = np.load(file_location + "/" + "/" + "aorc_{0}.npy".format(month))

    else:
        var_array = np.load(file_location + "/" + "/" + "{0}_{1}_aorc_res.npy".format(variable, month))

    if variable_short_name == 'mtpr':
        var_array = var_array * 3600 # convert the unit to mm

    full_aorc_index_list = np.arange(0, 1024 * 630)
    split_number = 1000
    sub_index_list = list(split_array(full_aorc_index_list, split_number))

    for batch_index in range(split_number):
        aorc_index_list = sub_index_list[batch_index]

        df_data = {}
        # for each grid 1140 * 630 = 718200 grids
        for i in aorc_index_list:

            # get the corresponding aorc grid index
            aorc_lat_index = i // 1024
            aorc_lon_index = i % 1024

            df_data[i] = var_array[:, aorc_lat_index, aorc_lon_index]

        var_df = pd.DataFrame(df_data)
        var_df = var_df.astype(np.float32) # conver the data to np.float32

        # create save location
        save_folder = r"/home/yliu2232/miss_design_storm/6h_training_df/full/{0}".format(month)
        create_folder(save_folder)

        # Create an directory
        create_folder(save_folder + "/" + "{0}".format((batch_index)))
        # save the training dataframe under the batch folder
        var_df.to_csv(save_folder + "/" + "{0}".format(batch_index) + "/" + "{0}_{1}.csv".format(batch_index, variable_short_name), index=False)


# creat a task, whcih is a dictionary containing year information
def create_task(month, variable_name, variable_short_name):
    task = {"month": month, 'variable':variable_name, 'short_name':variable_short_name}
    return task


def split_array(arr, parts):
    # used to split the array evenly
    length = len(arr)
    split_length = length // parts
    for i in range(parts):
        start = i * split_length
        end = start + split_length
        if i == parts - 1:
            end = length
        yield arr[start:end]



if __name__ == "__main__":

    process_num = 10
    month_list = [12, 1, 2, 3, 4, 5]

    variable_dict = {'mean_total_precipitation_rate': 'mtpr',
                     'total_column_water_vapour': 'tcwv',
                     'aorc': 'aorc',
                     'vertical_integral_of_water_vapour_flux': 'ivt'}

    task_list = []

    for month in month_list:
        for variable_name in variable_dict:
            # get the shortname
            variable_short_name = variable_dict[variable_name]
            task = create_task(month, variable_name, variable_short_name)
            task_list.append(task)

    # use the process
    pool = multiprocessing.Pool(processes = process_num)
    pool.map(generate_training_df, task_list)








