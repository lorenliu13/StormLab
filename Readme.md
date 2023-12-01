# StormLab: space-time nonstationary rainfall model for large area basins. 
<img src="/images/brief_figure_20231130.png" width="100%" height="100%">

This repository contains code to implement StormLab, a stochastic rainfall generator that simulate 6-hour, 0.03° resolution rainfall fields over large-area river basins conditioned on global climate model data. 

The main functions of the codes are as follows: 
1. Bias correction
	- Regrid ERA5 to CESM2 resolution. Bias-correct the CESM2 data against ERA5 using the CDF-t method. 
2. Storm tracking
	- Identify and track strong integrated water vapor transport events and associated rainstorms in ERA5 and CESM2 data. 
	- Extract atmospheric variable fields and AORC rainfall from identified ERA5 rainstorms.
	- Extract atmospheric variable fields from identified CESM2 rainstorms.  
3. Monthly distribution fitting
	- Fit distributions to ERA5 atmospheric variables and AORC rainfall at each grid cell. 
	- Generate fields for distribution parameter coefficients. 
4. AORC rainfall matching
	- Match AORC rainfall to CESM2 rainstorms. 
5. CESM2 random storm generation
	- Generate spatially correlated noise fields. 
	- Generate conditional distribution parameters based on CESM2 atmospheric variables. 
	- Simulate stochastic rainfall fields of CESM2 rainstorms. 

## Installation
`git clone https://github.com/lorenliu13/StormLab.git`
## Dependencies
Required packages are listed in requirements.txt
## Usage
1. Download the example data and unzip at the code folder "/StormLab."
	- Google Drive: https://drive.google.com/file/d/1MJzO8bhKJxZ5sc7OuIQQQPw3AmQLKSME/view?usp=sharing
	- The folder should look like: "/StormLab/data"
	- Running the code below will create an "/output" folder to save outputs in it. 
1. Rainstorm tracking
	- Perform storm tracking on ERA5 data
	  
		`python src/storm_tracking/era5_tracking/rainstorm_identification_and_tracking.py`
	- Generate ERA5 rainstorm catalog
	  
		`python src/storm_tracking/era5_tracking/generate_rainstorm_catalog.py`
	- Perform storm tracking on CESM2 data
	  
		`python src/storm_tracking/cesm_tracking/rainstorm_identification_and_tracking.py`
	- Generate CESM2 rainstorm catalog
	  
		`python src/storm_tracking/cesm_tracking/generate_rainstorm_catalog.py`
3. Generate ERA5/CESM2 atmospheric variable fields and AORC rainfall fields for identified rainstorms
	- Generate ERA5 atmospheric variable fields for ERA5 rainstorms
	
		`python src/era5_random_storms/extract_rainstorm_events/extract_era5_rainstorm_covariate_fields.py`
	- Generate AORC rainfall fields for ERA5 rainstorms
	  
		`python src/era5_random_storms/extract_rainstorm_events/extract_era5_rainstorm_rainfall_fields.py`
	- Generate CESM2 atmospheric variable fields for CESM2 rainstorms
	  
		`python src/cesm2_random_storms/extract_rainstorm_events/extract_cesm2_rainstorm_covariate_fields.py`
4. Generate ERA5-AORC dataframe at each grid cell and perform distribution fitting
	- Generate ERA5-AORC dataframe at each grid cell
	  
		`python src/era5_random_storms/monthly_distribution_fitting/generate_monthly_fitting_dataframe_at_each_grid.py`
	- Distribution fitting
	  
		`python src/era5_random_storms/monthly_distribution_fitting/fit_distribution_by_batch.py`
5. Simulate stochastic rainstorm based on CESM2 data
	- Generate 2D random Gaussian noise fields
	  
		`python src/cesm2_random_storms/noise_generation/rainstorm_noise_generation.py`
	- Generate parameter fields of conditional rainfall distributions
	  
		`python src/cesm2_random_storms/conditional_distribution_parameter_fields/distribution_param_field_for_rainstorm_event.py`
	- Generate stochastic rainstorm fields
	  
		`python src/cesm2_random_storms/random_rainfall_simulation/rainstorm_rainfall_simulation.py`
	
6. The other codes require a complete dataset to process and do not support example run.

## Code details:
The detailed code structure and functions are as follows: 

**Bias correction:**

1.1 ERA5 Regridding
- Code: src/bias_correction/era5_regrdding_to_cesm2.py
- Function: Generate 1979-2022 long series of ERA5 data at CESM2 resolution for target seasons. 
- Note: ERA5 mean total precipitation rate (mtpr) unit is mm/s, regrid method: conservative

1.2 CESM2 Data Extraction
- Code: src/bias_correction/extract_cesm2_seasonal_series.py
- Function: 
- Generate early (1950-1978), current (1979-2021), and late (2022-2050) periods of long series of CESM2 atmospheric variable array for four seasons. 
- Also generate a reference dataframe of year and month for each time step in the full array.

1.3 CDF-t Bias Correction
- Code: src/bias_correction/cesm2_bias_correction_by_era5.py
- Function: Bias-correct the long-term series of CESM2 variable array against ERA5 using the CDF-t method. 
- Note: 
	- CESM2 variable total precipitaiton (prect) output unit is mm.

1.4 Annual NetCDF Creation
- Code: src/bias_correction/generate_bias_corrected_annual_cesm2_netcdf.py
- Function: Split the long-term bias-corrected CESM2 into yearly array saved by netcdf files. 

**Storm tracking:**

2.1 ERA5 Tracking

2.1.1 ERA5 Regridding
- Code: src/storm_tracking/era5_tracking/generate_annual_era5_netcdf_at_cesm2_resolution.py
- Function: Regrid the yearly ERA5 data to CESM2 resolution before rainstorm tracking.

2.2.2 ERA5 Storm Tracking
- Code: src/storm_tracking/era5_tracking/rainstorm_identification_and_tracking.py
- Function: Identify and track strong integrated water vapor transport (IVT) event based on the ERA5 IVT data, and attach concurrent ERA5 precipitation. 

2.2.3 ERA5 Storm Catalog
- Code: src/storm_tracking/era5_tracking/generate_rainstorm_catalog.py
- Function: Create a storm catalog (dataframe) of identified rainstorm events. 

2.2.4 ERA5 Field Extraction
- Code: src/era5_random_storms/extract_rainstorm_events/extract_era5_rainstorm_covariate_fields.py
- Function: Generate ERA5 covariate fields for each identified rainstorm events.

2.2.5 AORC Field Extraction
- Code: src/era5_random_storms/extract_rainstorm_events/extract_era5_rainstorm_rainfall_fields.py
- Function: Generate AORC rainfall fields for each identified rainstorm events.

2.2.6 ERA5 Combination
- Code: src/era5_random_storms/extract_rainstorm_events/combine_all_rainstorm_era5_covariate_by_month.py
- Function: Combine the ERA5 covariate fields of all rainstorm events and regrid them to AORC resolution. This will be used to create dataframe at each grid cell for distribution fitting. 

2.2.7 AORC Combination
- Code: src/era5_random_storms/extract_rainstorm_events/combine_all_rainstorm_aorc_by_month.py
- Function: Combine the AORC rainfall fields of all rainstorm events. This will be used to create dataframe at each grid cell for distribution fitting. 

2.2 CESM2 Tracking

2.2.1 CESM2 Storm Tracking
- Code: src/storm_tracking/cesm_tracking/rainstorm_identification_and_tracking.py
- Function: Identify and track strong integrated water vapor transport (IVT) event based on the CESM2 IVT data, and attach concurrent CESM2 precipitation. 

2.2.2 CESM2 Storm Catalog 
- Code: src/storm_tracking/cesm_tracking/generate_rainstorm_catalog.py
- Function: Create a storm catalog (dataframe) of identified rainstorm events. 

2.2.3 CESM2 Field Extraction
- Code: src/cesm2_random_storms/extract_rainstorm_events/extract_cesm2_rainstorm_covariate_fields.py
- Function: Generate CESM2 covariate fields for each identified rainstorm events.

**Distribution Fitting**

3.1 Fitting Dataframe
- Code: src/era5_random_storms/monthly_distribution_fitting/generate_monthly_fitting_dataframe_at_each_grid.py
- Function: Generate a dataframe containing long-term series of ERA5 covariates or AORC rainfall at 1000 grid cells. This separates the 1024\*630 grids into batches of 1,000 grids. The fitting will be performed based on batches (loop through grids in a batch) rather than single grid by single grid to improve speed. 

3.2 Monthly Fitting
- Code: src/era5_random_storms/monthly_distribution_fitting/fit_distribution_by_batch.py
- Function: Load the dataframe for the current batch (containing 1,000 grids time series of ERA5 variables or AORC rainfall). Perform distribution fitting for each grid in the batch. 

3.3 Parameter Fields
- Code: src/era5_random_storms/monthly_distribution_fitting/generate_distribution_parameter_array.py
- Function: Generate the fitted distribution parameter coefficient fields.  

**AORC Rainfall Matching**

4.1 Field Creation for ERA5 and AORC
- Code: src/cesm2_random_storms/aorc_field_matching/create_long_term_era_fields.py
- Function: Create long-term 1979-2021 ERA5 covariate and AORC fields for matching

4.2 AORC Rainfall Matching
- Code: src/cesm2_random_storms/aorc_field_matching/match_aorc_rainfall.py
- Function: Sample the AORC fields for each CESM2 rainstorms based on k nearest neighbor method. 

**CESM2 Rainstorm Simulation**

5.1 Noise Generation
- Code: src/cesm2_random_storms/noise_generation/rainstorm_noise_generation.py
- Function: Generate space-time noise fields for each CESM2 rainstorm events.

5.2 Distribution Parameter Fields
- Code: src/cesm2_random_storms/conditional_distribution_parameter_fields/distribution_param_field_for_rainstorm_event.py
- Function: Generate conditional distribution parameter fields based on fitted coefficients and CESM2 large-scale atmospheric variable fields. 

5.3 Stochastic Rainfall Simulation
- Code: src/cesm2_random_storms/random_rainfall_simulation/rainstorm_rainfall_simulation.py
- Function: Generate simulated rainfall fields based on noise and conditional distribution parameter fields. 

## Citation
If you use this model in your work, please cite:
*Our publication is under preparation.*

## Contributing
Feel free to open an issue for bugs and feature requests.

## License
StormLab is released under the [MIT License](https://opensource.org/licenses/MIT).

## Authors
* [Yuan Liu](https://her.cee.wisc.edu/group-members/) - *research & developer*
* [Daniel B. Wright](https://her.cee.wisc.edu/group-members/) - *research*

## Attribution
This project uses code from the following repositories:
- [pysteps](https://pysteps.readthedocs.io/en/stable/)
- [STREAM](https://github.com/sam-hartke/STREAM)
- [starch](https://github.com/lorenliu13/starch/tree/master)