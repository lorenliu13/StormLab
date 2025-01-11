# Generate 2-D noise Gaussian noise field

import xarray as xr
import numpy as np
from pyproj import Transformer
import os
from scipy import interpolate


def _tukey(R, alpha):
    W = np.ones_like(R)
    N = min(R.shape[0], R.shape[1])

    mask1 = R < int(N / 2)
    mask2 = R > int(N / 2) * (1.0 - alpha)
    mask = np.logical_and(mask1, mask2)
    W[mask] = 0.5 * (
        1.0 + np.cos(np.pi * (R[mask] / (alpha * 0.5 * N) - 1.0 / alpha + 1.0))
    )
    mask = R >= int(N / 2)
    W[mask] = 0.0

    return W

def tukey_window_generation(m, n):
    X, Y = np.meshgrid(np.arange(n), np.arange(m))
    R = np.sqrt((X - int(n / 2)) ** 2 + (Y - int(m / 2)) ** 2)
    window_mask = _tukey(R, alpha = 0.2)
    # add small value to avoid zero
    window_mask += 1e-6
    return window_mask

def compute_amplitude_spectrum(field):
    # perform 2-d Fourier transform
    F = do_fft2(field)
    # normalize the imagery and real part
    F.imag = (F.imag - np.mean(F.imag)) / np.std(F.imag)
    F.real = (F.real - np.mean(F.real)) / np.std(F.real)
    # get the amplitude
    F_abs = np.abs(F)

    return F_abs

def do_fft2(array):
    return np.fft.fft2(array)

def do_ifft2(array):
    return np.fft.ifft2(array)

def FFST_based_noise_generation(field, noise, win_size, overlap, ssft_war_thr):
    """
    Generate noise field using FFST method based on rainfall field
    :param field: rainfall field
    :param noise: white noise
    :param win_size: window size (128, 128) tukey window
    :param overlap: overlap ratio of windows
    :param ssft_war_thr: wet area ratio for FFST, 0.1
    :return: correated noise field
    """

    dim_x = field.shape[1]  # get the column number 1100
    dim_y = field.shape[0]  # get the row number 630

    # number of windows
    num_windows_y = int(np.ceil(float(dim_y) / win_size[0]))
    num_windows_x = int(np.ceil(float(dim_x) / win_size[1]))

    # perform global FFT
    global_F = compute_amplitude_spectrum(field)
    noise_F = do_fft2(noise)  # get the white noise FFT field

    # get global noise field
    global_noise_array = do_ifft2(noise_F * global_F).real
    # final_noise_array = global_noise_array.copy()
    final_noise_array = np.zeros(global_noise_array.shape)
    final_weight_array = np.zeros(global_noise_array.shape)

    # loop rows
    for i in range(num_windows_y):
        # loop columns: this performs row-major looping, which is faster
        for j in range(num_windows_x):

            # prepare indices
            idxi = np.zeros(2).astype(int)
            idxj = np.zeros(2).astype(int)

            # compute indices of local window
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))  # get the upper y index
            idxi[1] = int(np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y)))  # get the lower y index

            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0))) # get the left x index
            idxj[1] = int(np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x)))  # get the right x index

            # for each window, get the subregion
            window_rainfall_array = field[idxi[0]: idxi[1], idxj[0]: idxj[1]]

            # get the mask
            curr_window_dimension = (idxi[1] - idxi[0], idxj[1] - idxj[0])
            tukey_window = tukey_window_generation(m=curr_window_dimension[0], n=curr_window_dimension[1])

            # get the wet area ratio as the portion of wet grid in the local window
            weighted_window_rainfall_array = window_rainfall_array * tukey_window
            wet_area_raito = np.sum((weighted_window_rainfall_array) > 0.01) / (
                        curr_window_dimension[0] * curr_window_dimension[1])

            # get the full masked rainfall field
            full_mask = np.zeros((dim_y, dim_x))
            full_mask[idxi[0]: idxi[1], idxj[0]: idxj[1]] = tukey_window

            if wet_area_raito > ssft_war_thr:

                # get masked rainfall fields
                full_masked_rainfall_array = field * full_mask

                # perform fourier transform to calculate the amplitude spectrum
                local_F = compute_amplitude_spectrum(full_masked_rainfall_array)

                # generate local noise
                local_noise_array = do_ifft2(noise_F * local_F).real

                # update the final noise field
                final_noise_array += local_noise_array * full_mask
                final_weight_array += full_mask

            else:
                # add global noise
                final_noise_array += global_noise_array * full_mask
                final_weight_array += full_mask

    # compute the final noise as weighted average
    final_noise_array[final_weight_array > 0] /= final_weight_array[final_weight_array > 0]
    # normalize it
    final_noise_array = (final_noise_array - np.mean(final_noise_array)) / np.std(final_noise_array)

    return final_noise_array


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


def coordinate_transform(lon_data, lat_data):
    """
    Transform geographic coordinate (degree) to  projected coordinate (m)
    :param lon_data: Longitude coordinate of the grid (degree)
    :param lat_data: Latitude coordinate of the grid (degree)
    :return:
    """
    lon_array, lat_array = np.meshgrid(lon_data, lat_data)
    lon_array_flat = lon_array.flatten()
    lat_array_flat = lat_array.flatten()
    points = []
    for i in range(lon_array_flat.shape[0]):
        points.append((lat_array_flat[i], lon_array_flat[i]))
    # transform lat lon to projected coordinate system
    transformer = Transformer.from_crs(4326, 2163)
    lat_prj_array = []
    lon_prj_array = []
    for pt in transformer.itransform(points):
        lat_prj_array.append(pt[1])
        lon_prj_array.append(pt[0])
    lat_prj_array = np.reshape(np.array(lat_prj_array), lat_array.shape)  # projected location in m
    lon_prj_array = np.reshape(np.array(lon_prj_array), lat_array.shape)  # projected location in m

    return lat_prj_array, lon_prj_array


# Compute the lag-1 correlation of the rainfall field
def lag1_autocorr(matrix):
    # Subtract the mean along the first axis
    demeaned_matrix = matrix - matrix.mean(axis=0, keepdims=True)

    # Compute the product of the matrix with its lagged version (shifted along the first axis)
    lagged_product = demeaned_matrix[:-1] * demeaned_matrix[1:]

    # Compute the sums needed for the auto-correlation coefficients
    numerator = np.sum(lagged_product, axis=0)
    denominator = np.sqrt(np.sum(demeaned_matrix[:-1] ** 2, axis=0) * np.sum(demeaned_matrix[1:] ** 2, axis=0))

    # set where denominator = 0 to 1
    numerator[denominator == 0] = 1
    denominator[denominator == 0] = 1
    # Compute the auto-correlation coefficients, handling the case when the denominator is 0
    autocorr_coeffs = numerator / denominator

    return autocorr_coeffs

def temporal_autocorrelation(prcp_array):
    """
    Calculate the lag-1 auto-correlation coefficient from the input precipitation field.
    :param prcp_array: coarse-scale precipitation data from global climate models, dimension (time, lat, lon), unit: mm/h
    :return:
    """

    # compute the temporal auto-correlation of the rainfall fields (using the raw precipitation array at CESM2 resolution)
    # flat the precipitation array along x and y axis, [time, y, x] into [time, y * x]
    flat_prcp_vector = np.reshape(prcp_array, (prcp_array.shape[0], -1))
    # compute pearson correlation
    alpha_list = []
    alpha_list.append(0.5)  # append an initial value of 0.5, this value is not used, just for adding one step here
    for i in range(prcp_array.shape[0] - 1):
        # get rid of purely dry girds, i.e., those grids have no rain at both time step
        vector_mask = (flat_prcp_vector[i] != 0) | (flat_prcp_vector[i + 1] != 0)
        prev_rain_vector = flat_prcp_vector[i][vector_mask]
        curr_rain_vector = flat_prcp_vector[i + 1][vector_mask]

        corr = np.corrcoef(prev_rain_vector, curr_rain_vector)  # compute pearson correlation coefficient [2,2] matrix
        alpha_list.append(corr[0, 1])  # get the coefficient, which is located at [0, 1]
    alpha_list = np.array(alpha_list)

    return alpha_list

def noise_generation(prcp_array, acf_array, u_array, v_array, lon_data, lat_data, window_size,
                     overlap_ratio, ssft_war_thr, seed):
    """
    # Generating 2D Gaussian noise field based on input precipitation, wind speed fields.
    # Note: Wind advection is generally recommended when using hourly or 3-hour wind field.
    # For time interval >= 6-hours, wind advection is not recommended because assuming constant wind speed for larger than 6 hours may lead to distortion of the noise field.
    # In this case, the wind field can be set as zero.
    :param prcp_array: high-resolution precipitation field data, dimension (time, lat, lon), unit: mm/h
    :param acf_array: lag-1 auto-correlation coefficient array (time, lat, lon), unit: dimensionless
    :param u_array: high-resolution u wind field data, dimension (time, lat, lon), unit: m/h
    :param v_array: high-resolution v wind field data, dimension (time, lat, lon), unit: m/h
    :param lon_data: longitude coordinate of the high-resolution precipitation field, dimension (lon,) (degree)
    :param lat_data: latitude coordinate of the high-resolution precipitation field, dimension (lat,) (degree)
    :param window_size: size of the local fft window, default: 128
    :param window_function: the tapering function for local fft window, default: tukey
    :param overlap_ratio: overlap ratio of each local fft window, default: 0.3
    :param ssft_war_thr: threshold of wet area ratio to perform local fft, default: 0.1
    :param seed: random seed for noise generation
    :return:
    """

    # First, we calculate the length (in m) of each grid cell for noise advection
    # get projected coordinates in the unit of m
    lat_prj_array, lon_prj_array = coordinate_transform(lon_data, lat_data)
    # compute the distance between grids x[i+1] - x[i] (unit: m)
    x_diff = np.diff(lon_prj_array, axis=1)
    # append a column at the begining column
    x_diff = np.insert(x_diff, 0, x_diff[:, 0], axis=1)
    # compute the distance between grids y[i] - y[i+1] (unit: m), note that y[i+1] < y[i], so there is a negative term
    y_diff = -np.diff(lat_prj_array, axis=0)
    # append a row at the begining
    y_diff = np.insert(y_diff, 0, y_diff[0, :], axis=0)
    # get the precipitation field size
    x_size = lon_data.shape[0]
    y_size = lat_data.shape[0]
    # get x and y index array
    yind = np.repeat(np.arange(0, y_size), x_size).reshape(y_size, x_size).astype('int16')  # get an array of y index
    xind = np.tile(np.arange(0, x_size), y_size).reshape(y_size, x_size).astype('int16')  # get an array of x index


    # initialize a zero field with dimension (time, lat, lon) to store the final noise field
    final_prcp_noise_array = np.zeros(prcp_array.shape)

    # generate random Gaussian noise with the same dimension as the rainfall array
    randstate = np.random
    randstate.seed(seed)
    raw_noise_field = randstate.randn(prcp_array.shape[0], prcp_array.shape[1], prcp_array.shape[2])

    for time_step in range(prcp_array.shape[0]):

        # at the first time step, the noise is generated without advection
        if time_step == 0:
            # generate correlated random noise
            final_prcp_noise_array[time_step] = FFST_based_noise_generation(field=prcp_array[time_step],
                                                                            noise=raw_noise_field[time_step],
                                                                            win_size=window_size,
                                                                            overlap=overlap_ratio,
                                                                            ssft_war_thr=ssft_war_thr)

        else: # starting from the second time step, we need to compute noise advection

            dis_x_grid = np.round(u_array[time_step] / x_diff)  # compute grid displacement of x
            dis_y_grid = np.round(v_array[time_step] / y_diff)  # compute grid displacement of y

            x_advected_index = np.array((xind - dis_x_grid), dtype='int')  # the advected index of x
            x_out_boundary_mask = (x_advected_index < 0) | (
                    x_advected_index > (x_size - 1))  # True if the element is currently empty at current step
            # get the correct index
            x_advected_index_scaled = x_advected_index % x_size

            y_advected_index = np.array((yind - dis_y_grid), dtype='int')
            y_out_boundary_mask = (y_advected_index < 0) | (
                    y_advected_index > (y_size - 1))  # True if the element is empty at current step
            y_advected_index_scaled = y_advected_index % y_size

            # get the noise field at t-1
            previous_noise_field = final_prcp_noise_array[time_step - 1]
            # get the first part of the current noise field at time t
            advected_noise = previous_noise_field[y_advected_index_scaled, x_advected_index_scaled]  # advection

            # get current noise field
            current_noise_field = FFST_based_noise_generation(field = prcp_array[time_step],
                                                                            noise = raw_noise_field[time_step],
                                                                            win_size = window_size,
                                                                            overlap = overlap_ratio,
                                                                            ssft_war_thr = ssft_war_thr)

            # fill the empty element with current correlated noise field
            advected_noise = np.where((x_out_boundary_mask) | (y_out_boundary_mask), current_noise_field, advected_noise)

            # compute the final noise
            final_prcp_noise_array[time_step] = acf_array[time_step] * advected_noise + np.sqrt(
                1 - acf_array[time_step] ** 2) * current_noise_field

    # step_report['alpha'] = np.array(alpha_list)
    return final_prcp_noise_array



if __name__ == "__main__":

    seed = 1
    ar_id = 202200018

    cesm_rainstorm_folder = r"C:\Disk_D\My_drive\Code\My_open_source\StormLab\data\cesm2\cesm2_rainstorm_covariates"
    match_aorc_rainfall_folder = r"C:\Disk_D\My_drive\Code\My_open_source\StormLab\data\cesm2\matched_aorc_rainfall"

    # load precipitation data
    raw_cesm_prcp_xarray = xr.load_dataset(cesm_rainstorm_folder + "/" + "{0}_prect_cesm_res.nc".format(ar_id))
    raw_cesm_prcp_array = raw_cesm_prcp_xarray['prect'].data

    # calculate the correlation
    auto_corr_list = temporal_autocorrelation(raw_cesm_prcp_array)

    # load precipitation data
    aorc_array = np.load(match_aorc_rainfall_folder + "/" + "{0}_sr_rainfall.npy".format(ar_id))
    # load wind data
    u_xarray = xr.load_dataset(cesm_rainstorm_folder + "/" + "{0}_u850_cesm_res.nc".format(ar_id))
    v_xarray = xr.load_dataset(cesm_rainstorm_folder + "/" + "{0}_v850_cesm_res.nc".format(ar_id))

    # get numpy array
    time_steps = u_xarray['time'].data
    # aorc_array = aorc_xarray['aorc'].data
    coarse_u_array = u_xarray['u850'].data
    coarse_v_array = v_xarray['v850'].data

    u_array = np.zeros((coarse_u_array.shape[0], 630, 1024))
    v_array = np.zeros((coarse_v_array.shape[0], 630, 1024))

    # set up AORC coordinates
    aorc_lat = np.linspace(50, 29, 630)
    aorc_lon = np.linspace(-113.16734, -79.068704, 1024)

    cesm_lat = np.linspace(50.41884817, 28.7434555, 24)
    cesm_lon = np.linspace(-113.75, -78.75, 29)

    # interpolate to AORC resolution
    for time_index in range(coarse_u_array.shape[0]):
        curr_coarse_u_array = coarse_u_array[time_index]
        curr_coarse_v_array = coarse_v_array[time_index]

        u_array[time_index] = linear_interpolation(curr_coarse_u_array, cesm_lon, cesm_lat, aorc_lon, aorc_lat)
        v_array[time_index] = linear_interpolation(curr_coarse_v_array, cesm_lon, cesm_lat, aorc_lon, aorc_lat)

    # set up AORC coordinates
    aorc_lat = np.linspace(50, 29, 630)
    aorc_lon = np.linspace(-113.16734, -79.068704, 1024)
    # replace -29997 with 0
    # aorc_array = np.where(aorc_array < 0.2, 0, aorc_array)



    window_size = (128, 128)
    window_function = 'tukey'
    overlap_ratio = 0.3
    ssft_war_thr = 0.1

    # generate a copy
    sub_ar_prcp_field_copy = np.where(aorc_array < 0.2, 0, aorc_array)

    final_prcp_noise_array = noise_generation(sub_ar_prcp_field_copy, auto_corr_list, u_array*3600*6, v_array*3600*6, aorc_lon, aorc_lat, window_size,
                     overlap_ratio, ssft_war_thr, seed)
