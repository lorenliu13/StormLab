# Regrid the yearly ERA5 data to CESM2 resolution before rainstorm tracking.
# Yuan Liu
# 2023/07/17
import xarray as xr
import numpy as np
import xesmf as xe



if __name__ == "__main__":

    year_list = np.arange(1979, 2022)

    variable_list = ["850_u_component_of_wind", "850_v_component_of_wind"]

    era5_lon = np.linspace(-114, -78, 145)
    era5_lat = np.linspace(51, 28, 93)

    cesm_lon = np.linspace(-113.75, -78.75, 29)
    cesm_lat = np.linspace(50.41884817, 28.7434555, 24)


    for variable in variable_list:
        # get an array to store data
        full_length_era_array = []
        for year in year_list:
            print("Start to process {0} {1}".format(variable, year))
            # load precipitation dataset
            era5_xarray = xr.load_dataset(r"/home/yliu2232/miss_design_storm/raw_data/ERA5/ERA5_6h/{0}/ERA5_6H_{1}_{0}.nc".format(year, variable))
            # get the variable short name
            short_name = [k for k in era5_xarray.data_vars.keys()]
            short_name = short_name[0]

            target_grid = xr.Dataset({
                'latitude': (('latitude',), cesm_lat),
                'longitude': (('longitude',), cesm_lon)
            })

            if variable == 'mean_total_precipitation_rate':
                regridder = xe.Regridder(era5_xarray, target_grid, method='conservative')
            else:
                regridder = xe.Regridder(era5_xarray, target_grid, method='bilinear')

            # upscaled to CESM resolution
            era5_coarse_ds = regridder(era5_xarray, keep_attrs=True)  # get the regrided dataset

            era5_coarse_ds.to_netcdf(r"/home/yliu2232/miss_design_storm/raw_data/ERA5/ERA5_6h/{0}".format(year) + "/" + "ERA5_6H_{0}_{1}_cesm_res.nc".format(variable, year),
                                  encoding={short_name: {"dtype": "float32", "zlib": True}})


