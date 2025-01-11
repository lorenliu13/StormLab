# The storm tracker that tracks storm along the time steps.
# 2021/12/21


import copy
from math import sqrt
import numpy as np
from skimage.segmentation import relabel_sequential
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure
from skimage import morphology
from skimage import draw
import skimage
import os


def track(grown_array: np.ndarray, ratio_threshold: float, dry_spell_time: int):
    """
    Storm tracking method that labels consecutive storms over time with the same integer labels. The code is modified
    based on github project Storm Tracking and Evaluation Protocol (https://github.com/RDCEP/STEP,
    author: Alex Rittler.
    :param grown_array: Result array from storm identification with dimension of (time, lon, lat).
    :param prcp_array: Raw precipitation field with dimension of (time, lon, lat).
    :param ratio_threshold: Threshold of overlapping ratio, default is 0.3.
    :param dry_spell_time: Allow method to match storm at the time step of (t-1-dry_spell_time), if no match is found at
    t-1 step, default is 0.
    :return:
    """

    # get total time slice
    num_time_slices = grown_array.shape[0]
    # make a copy of the result of the identification algorithm to avoid labeling collisions
    # we will record any labeling changes here
    result_data = copy.deepcopy(grown_array)

    # skip labeling t=0, since it is already labeled correctly
    # for every other time slice
    for time_index in range(1, num_time_slices):

        # find the labels for this time index and the labeled storms in the previous time index
        current_labels = np.unique(grown_array[time_index])

        # and prepare the corresponding precipitation data
        # curr_precip_data = prcp_array[time_index]

        # determine the maximum label already used to avoid collisions
        if time_index == 1:
            max_label_so_far = np.max(result_data[time_index])
        else:
            max_label_so_far = np.max(result_data[:time_index])
        # print time index
        # print("Time slice : {0}".format(time_index))
        # then, for each label in current time index (that isn't the background)
        for label in current_labels:
            if label:
                # print current storm number
                # print(f'Current storm label {0}'.format(label))
                # make sure initially the max storm size and best matched storm are 0
                max_size = 0
                best_matched_storm = 0
                # find where the labels of the current storm segment exist in the current time slice
                current_label = np.where(grown_array[time_index] == label, 1, 0)
                curr_size = np.sum(current_label)

                # find the precipitation data at those locations
                # curr_label_precip = np.where(grown_array[time_index] == label, curr_precip_data, 0)

                # and its intensity weighted centroid
                # curr_centroid = center_of_mass(curr_label_precip)

                # match storms at forward time steps
                if time_index >= dry_spell_time + 1:
                    # back_step = 1, 2 if dry_spell_time = 1
                    for back_step in np.arange(1, dry_spell_time + 2):
                        # print("Match previous storm at {0}".format(time_index - back_step))
                        max_size, best_matched_storm = storm_match(result_data, max_size,
                                                                   best_matched_storm, time_index, back_step,
                                                                   current_label, curr_size,
                                                                   ratio_threshold)
                        # if find a match, stop current loop
                        if max_size:
                            break
                else:
                    # if time_index < dry_spell_time
                    back_step = 1
                    max_size, best_matched_storm = storm_match(result_data, max_size,
                                                               best_matched_storm, time_index, back_step,
                                                               current_label, curr_size,
                                                               ratio_threshold)
                # if we found matches
                if max_size:
                    # link the label in the current time slice with the appropriate storm label in the previous
                    result_data[time_index] = np.where(grown_array[time_index] == label, best_matched_storm,
                                                       result_data[time_index])

                # otherwise we've detected a new storm
                else:
                    # give the storm a unique label
                    result_data[time_index] = np.where(grown_array[time_index] == label, max_label_so_far + 1,
                                                       result_data[time_index])

                    max_label_so_far += 1

    result_data = result_data.astype('int')
    seq_result = relabel_sequential(result_data)[0]

    return seq_result


def displacement(current: np.ndarray, previous: np.ndarray) -> np.array:
    """Computes the displacement vector between the centroids of two storms.
    :param current: the intensity-weighted centroid of the storm in the current time slice, given as a tuple.
    :param previous: the intensity-weighted centroid of the storm in the previous time slice, given as a tuple.
    :return: the displacement vector, as an array.
    """
    return np.array([current[0] - previous[0], current[1] - previous[1]])


def magnitude(vector: np.ndarray) -> float:
    """Computes the magnitude of a vector.
    :param vector: the displacement vector, given as an array.
    :return: its magnitude, as a float.
    """
    return sqrt((vector[0] ** 2) + (vector[1] ** 2))


def storm_match(result_data : np.ndarray, max_size : float,
                best_matched_storm : int, time_index : int, back_step : int, current_label : int,
                curr_size : int, ratio_threshold : float):
    """
    The algorithm that searches the best match previous storm for the current storm.
    :param result_data: Storm identification array.
    :param prcp_array: Raw precipitation array.
    :param max_size: Current matched storm size.
    :param best_matched_storm: ID of the current best matched storm.
    :param time_index: Current time step.
    :param back_step: Backward step number for storm match. Previous time step = time_index - back_step.
    :param current_label: Label of the current storm.
    :param curr_size: Size of the current storm in pixels.
    :param curr_centroid: Centroid of the current storm.
    :param ratio_threshold: Threshold of overlapping ratio
    :return:
    max_size: The size of the best matched storm.
    best_matched_storm: The label of the best matched storm.
    """
    max_ratio = 0
    prev_size = 0
    # get previous storm ids and prcp data
    previous_storms = np.unique(result_data[time_index - back_step])
    # prev_precip_data = prcp_array[time_index - back_step]

    for storm in previous_storms:
        if storm == 0:  # skip the background
            continue

        # find the storm location in previous time step
        previous_storm = np.where(result_data[time_index - back_step] == storm, 1, 0)
        prev_size = np.sum(previous_storm)

        # selected the overlap area of current storm to prev storm
        overlap_curr_to_prev = np.where(previous_storm == 1, current_label, 0)

        # compute overlapping size
        overlap_size_curr_to_prev = np.sum(overlap_curr_to_prev)
        # compute the overlapping ratio A/current_storm_size
        overlap_ratio_curr_to_prev = overlap_size_curr_to_prev / curr_size

        # selected the overlap area of prev to curr
        overlap_prev_to_curr = np.where(current_label == 1, previous_storm, 0)

        overlap_size_prev_to_curr = np.sum(overlap_prev_to_curr)
        # compute the overlapping ratio: A/previous_storm_size
        overlap_ratio_prev_to_curr = overlap_size_prev_to_curr / prev_size

        # add the two ratio together = A/current_storm_size + A/previous_storm_size
        integrated_ratio = overlap_ratio_curr_to_prev + overlap_ratio_prev_to_curr

        # find the largest overlapping ratio
        if integrated_ratio > max_ratio:
            max_ratio = integrated_ratio
            temp_matched_storm = storm
    # if the max overlapping ratio is larger than threshold
    if max_ratio > ratio_threshold:

        # prev_storm_precip = np.where(result_data[time_index - back_step] == temp_matched_storm, prev_precip_data, 0)
        # prev_centroid = center_of_mass(prev_storm_precip)
        # curr_prev_displacement = displacement(curr_centroid, prev_centroid)  # compute displacement vector
        # curr_prev_magnitude = magnitude(curr_prev_displacement)  # compute centroid distance in pixel
        # if curr_prev_magnitude < max_distance:
        best_matched_storm = temp_matched_storm
        max_size = prev_size

    return max_size, best_matched_storm


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
    # print("Attach precipitation data to AR in  {0}".format(year))

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