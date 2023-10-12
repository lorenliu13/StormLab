# Identify and track strong integrated water vapor transport (IVT) event based on the CESM2 IVT data, and attach concurrent CESM2 precipitation.
# Yuan Liu
# 05/09/2023

import copy
import numpy as np
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure
from skimage.segmentation import relabel_sequential
from skimage import morphology
from skimage import draw
import skimage
import xarray as xr
from tracking_ar import track
import os


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error

        pass


def perform_connected_components(to_be_connected: np.ndarray,
                                 connectivity_type: np.ndarray) -> None:
    """Higher order function used to label connected-components on all time slices of a dataset.
    :param to_be_connected: the data to perform the operation on, given as an array of dimensions Time x Rows x Cols.
    :param result: where the result of the operation will be stored, with the same dimensions as to_be_connected.
    :param lifetime: the number of time slices in the data, given as an integer.
    :param connectivity_type: an array representing the type of connectivity to be used by the labeling algorithm. See
    scipy.ndimage.measurements.label for more information.
    :return: (None - the operation is performed on the result in place.)
    """
    # for index in range(lifetime):
    cc_output, _ = label(to_be_connected, connectivity_type)  # label also returns # of labels found

    return cc_output


def perform_morph_op(morph_function: object, to_be_morphed: np.ndarray,
                     structure: np.ndarray) -> None:
    """Higher order function used to perform a morphological operation on all time slices of a dataset.
    :param morph_function: the morphological operation to perform, given as an object (function).
    :param to_be_morphed: the data to perform the operation on, given as an array of dimensions Time x Rows x Cols.
    :param result: where the result of the operation will be store, with the same dimensions as to_be_morphed.
    :param lifetime: the number of time slices in the data, given as an integer.
    :param structure: the structural set used to perform the operation, given as an array. See scipy.morphology for more
    information.
    :return: (None - the operation is performed on the result in place.)
    """
    operation = morph_function(to_be_morphed, structure)

    return operation


def build_morph_structure(radius: int):
    """
    Create an array for morphological operation
    :param radius: the radius of the circle area in the array, where in the circle the element is 1, outside the circle the element is 0
    :return: an array for morphological operation
    """
    struct = np.zeros((2 * radius, 2 * radius))
    rr, cc = draw.disk(center=(radius - 0.5, radius - 0.5), radius=radius)
    struct[rr, cc] = 1  # data in the circle equals 1
    return struct


def ivt_identification(ivt_data: np.ndarray, morph_radius: int,
                       high_threshold: float, low_threshold: float, expand_distance: int):
    """
    Identify AR event as strong integrated water vapour flux transport (IVT) area based on ERA5 reanalysis data
    :param ivt_data: integrated water vapour flux array with dimension (lat, lon)
    :param morph_radius: the radius of the morphological structure for IVT identification
    :param high_threshold: a high threshold to identify the high IVT area
    :param low_threshold: a low threshold to expand the identified high IVT area
    :param expand_distance: the maximum distance when expanding the high IVT area to low threshold
    :return: an array with dimension (lat, lon), the array elements are labelled by an unique integers if an AR is identified
    """

    # define a morph structure
    morph_structure = build_morph_structure(radius=morph_radius)
    # filter with high threhsold
    high_threshold_data = np.where(ivt_data >= high_threshold, ivt_data, 0)
    low_threshold_data = np.where(ivt_data >= low_threshold, ivt_data, 0)

    # use 8-connectivity for determining connectedness below
    connectivity = generate_binary_structure(2, 2)

    # run the connected-components algorithm on the data and store it in our new array
    # since we will repeat this process often, we use a higher order function to compute the desired result
    label_array = perform_connected_components(high_threshold_data, connectivity)

    # perform a morph dialation
    filtered_label_array = perform_morph_op(morphology.dilation, label_array, morph_structure)
    # perform connected labeling to merge close components
    filtered_label_array = perform_connected_components(filtered_label_array, connectivity)
    # apply it to raw labelled array
    processed_label_array = np.where((label_array != 0), filtered_label_array, 0)

    # plt.figure()
    # plt.pcolormesh(ivt_lon, ivt_lat, np.ma.masked_where(processed_label_array == 0, processed_label_array), cmap = 'hsv')

    # perform a morph erosion
    # make a copy
    temp_label_array = copy.deepcopy(processed_label_array)
    eroded_label_array = perform_morph_op(morphology.erosion, temp_label_array, morph_structure)
    # keep only those storm ids in eroded_labeled_array
    unique_storm_labels = np.unique(eroded_label_array)
    unique_storm_labels = unique_storm_labels[unique_storm_labels != 0]
    processed_label_array = np.where(np.isin(processed_label_array, unique_storm_labels), processed_label_array, 0)

    # plt.figure()
    # plt.pcolormesh(ivt_lon, ivt_lat, np.ma.masked_where(processed_label_array == 0, processed_label_array), cmap = 'hsv')

    # grow to lower boundary
    expanded_label_array = skimage.segmentation.expand_labels(processed_label_array, distance=expand_distance)
    # intersect with low precipitation boundary
    grown_label_array = np.where(low_threshold_data != 0, expanded_label_array, 0)

    return grown_label_array


def attach_prcp(track_array, prcp_array):
    """
    Attached the associated precipitation event to each tracked AR event.
    :param track_array: AR tracking result array with dimension (time, lat, lon)
    :param prcp_array: ERA5 precipitation array with dimension (time, lat, lon)
    :return: an array of (time, lat, lon), where the AR precipitation area is labelled with the same id as the tracked ar event
    """

    prcp_label_list = []
    print("Attach precipitation data to AR in  {0}".format(year))

    # for each time step
    for time_index in range(track_array.shape[0]):

        prcp_data = prcp_array[time_index]
        track_data = track_array[time_index]
        # threshold precipitation by 0.1 mm/h
        filtered_prcp = np.where(prcp_data < 0.1, 0, prcp_data)
        connectivity = generate_binary_structure(2, 2)
        prcp_label_array = perform_connected_components(filtered_prcp, connectivity)
        # if elements in prcp_label_array has overlap with the AR covered area, keep that precipitation region
        # add an number to the connected precipitation array to avoid index overlapping
        prcp_label_array = np.where(prcp_label_array != 0, prcp_label_array + np.max(track_data), 0)
        # find all the unique ar id in ar track data at this time step
        track_ar_ids = np.unique(track_data)
        track_ar_ids = track_ar_ids[track_ar_ids != 0] # remove zero because it is tbe background value
        id_pixel_counts = []
        for track_ar_id in track_ar_ids:
            # count the number of pixels belonging to the current ar id
            counts = np.sum(track_data == track_ar_id)
            id_pixel_counts.append(counts)
        # this vector contains the area of each ar event at this time step (unit: pixels)
        id_pixel_counts = np.array(id_pixel_counts)

        # sort from small to large, return the index
        id_index_seq = np.argsort(id_pixel_counts)
        # for AR area from small to large, assign contiguous precipitation regions to the AR
        # this means if there are two AR with common precipitation area, the precipitation will be labeled
        # to the AR having larger area.
        for index in id_index_seq:
            # get an ar event id, the one with larger area will come first
            track_ar_id = track_ar_ids[index]
            # find all the isolated precipitation regions intersect with AR
            overlap_labels = np.unique(np.where(track_data == track_ar_id, prcp_label_array, 0))
            overlap_labels = overlap_labels[overlap_labels != 0]
            # change the precipitation label to track ar id
            prcp_label_array = np.where(np.isin(prcp_label_array, overlap_labels), track_ar_id, prcp_label_array)

        # change other region to zero
        prcp_label_array = np.where(np.isin(prcp_label_array, track_ar_ids), prcp_label_array, 0)

        # append to the list
        prcp_label_list.append(prcp_label_array)
    # convert the last to numpy array
    prcp_label_array = np.array(prcp_label_list)

    return prcp_label_array



if __name__ == "__main__":

    ensemble_year = 1251
    # ensemble_id = 12
    # ensemble_ids = np.arange(11, 21)
    ensemble_ids = np.arange(16, 21)

    for ensemble_id in ensemble_ids: # Loop through ensemble members
        print("Start ensemble_id {0}".format(ensemble_id))
        year_list = np.arange(1950, 2051)
        # year_list = np.arange(2016, 2017)

        # year_list = [2020]
        for year in year_list:
            # save location
            save_folder = r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_tracking_bs" + "/" + "{0}_{1}".format(ensemble_year, ensemble_id)

            # load 3-hour ERA5 integrated water vapour flux (IVT) data with dimension (time, lat, lon)
            # ivt_xarray = xr.open_dataset(r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour/{0}_{1}/{2}".format(ensemble_year, ensemble_id, year) + "/" + "CESM2_ivt_{0}.nc".format(year))
            ivt_xarray = xr.open_dataset(
                r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/bias_corrected_series/{0}_{1}/{2}".format(ensemble_year,
                                                                                                   ensemble_id,
                                                                                                   year) + "/" + "CESM2_{0}_ivt_bs.nc".format(
                    year))
            ivt_array = ivt_xarray['ivt'].data

            # set up idnetification and tracking parameters
            morph_radius = 1 # 0.25 degree for 4, 1.25 degree for 1
            high_threshold = 500
            low_threshold = 250
            expand_distance = 5

            # create an empty array
            identification_array = np.zeros(ivt_array.shape)
            print("Start AR identification in {0}".format(year))
            # for each time step, perform ar identification on the IVT field
            for time_index in range(ivt_array.shape[0]):
                ivt_data = ivt_array[time_index, :, :]
                grown_label_array = ivt_identification(ivt_data, morph_radius, high_threshold, low_threshold, expand_distance)
                # relabel
                grown_label_array = relabel_sequential(grown_label_array)[0]
                # update empty array
                identification_array[time_index] = grown_label_array

            # save the identified array
            identification_array = identification_array.astype('int')
            # create folder
            create_folder(save_folder)
            np.save(save_folder + "/" + "{0}_identification.npy".format(year), identification_array)

            # load storm identification array
            # identification_array = np.load(save_folder + "/" + "{0}_identification.npy".format(year))

            # run ar tracking code
            print("Start AR tracking in {0}".format(year))
            track_array = track(identification_array, ratio_threshold=0.2, dry_spell_time=0)

            # change the data type to int
            track_array = track_array.astype('int')
            # save tracking array
            np.save(save_folder + "/" + "{0}_tracking.npy".format(year), track_array)

            # load 3-hour ERA5 precipitation data
            # prcp_xarray = xr.open_dataset(r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour/{0}_{1}/{2}".format(ensemble_year, ensemble_id, year) + "/" + "CESM2_prect_{0}.nc".format(year))
            prcp_xarray = xr.open_dataset(
                r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour_bias_correction/bias_corrected_series/{0}_{1}/{2}".format(ensemble_year,
                                                                                                   ensemble_id,
                                                                                                   year) + "/" + "CESM2_{0}_prect_bs.nc".format(
                    year))
            prcp_array = prcp_xarray['prect'].data # default unit: mm

            # attach associated precipitation event to each ar event
            prcp_label_array = attach_prcp(track_array, prcp_array)
            np.save(save_folder + "/" + "{0}_attached_prcp.npy".format(year), prcp_label_array)
