
import numpy as np
from skimage.segmentation import relabel_sequential
from .tracking_utility import track
from .tracking_utility import ivt_identification
from .tracking_utility import attach_prcp
# import xarray as xr


def rainstorm_tracking(ivt_array, low_threshold=250, high_threshold=500, morph_radius=1, expand_distance=5, overlap_ratio=0.2, dry_spell_time=0):

    # create an empty array
    identification_array = np.zeros(ivt_array.shape)

    # for each time step, perform ar identification on the IVT field
    for time_index in range(ivt_array.shape[0]):
        ivt_data = ivt_array[time_index, :, :]
        grown_label_array = ivt_identification(ivt_data, morph_radius, high_threshold, low_threshold, expand_distance)
        # relabel
        grown_label_array = relabel_sequential(grown_label_array)[0]
        # update empty array
        identification_array[time_index] = grown_label_array

    identification_array = identification_array.astype('int')

    track_array = track(identification_array, ratio_threshold=overlap_ratio, dry_spell_time=dry_spell_time)

    # change the data type to int
    track_array = track_array.astype('int')

    return track_array


def attach_precipitation(prcp_array, track_array):

    prcp_label_array = attach_prcp(track_array, prcp_array)

    return prcp_label_array




if __name__ == "__main__":

    print('TEST CODE')

    import xarray as xr

    cesm_folder = r"C:\Disk_D\My_drive\Code\My_open_source\StormLab\data\cesm2\bias_corrected_annual_cesm2\1251_18\2022"

    # load 3-hour CESM2 integrated water vapour flux (IVT) data with dimension (time, lat, lon)
    ivt_xarray = xr.open_dataset(cesm_folder + "/" + "CESM2_2022_ivt_bs.nc")
    ivt_array = ivt_xarray['ivt'].data

    # set up idnetification and tracking parameters
    morph_radius = 1  # 0.25 degree for 4, 1.25 degree for 1
    high_threshold = 500
    low_threshold = 250
    expand_distance = 5


    track_array = rainstorm_tracking(ivt_array, low_threshold=250, high_threshold=500, morph_radius=1, expand_distance=5,
                       overlap_ratio=0.2, dry_spell_time=0)

    # prcp_xarray = xr.open_dataset(r"/home/yliu2232/miss_design_storm/processed_data/cesm2/6_hour/{0}_{1}/{2}".format(ensemble_year, ensemble_id, year) + "/" + "CESM2_prect_{0}.nc".format(year))
    prcp_xarray = xr.open_dataset(
        cesm_folder + "/" + "CESM2_2022_prect_bs.nc")
    prcp_array = prcp_xarray['prect'].data  # default unit: mm

    # attach associated precipitation event to each ar event
    prcp_label_array = attach_prcp(track_array, prcp_array)