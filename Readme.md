# StormLab: Space-time Nonstationary Rainfall Model for Large Area Basins 
<img src="/images/brief_figure_20231130.png" width="100%" height="100%">
Figure: (A) Graphical showing how StormLab converts large-scale climate model predictions into thosands of realistic high-resolution rainfall simulations. (B) Rainfall "frequency curve" for Mississippi Basin rainfall, showing close agreement between high-resolution StormLab simulations and validation measurements.

## Introduction
This repository contains code to implement StormLab, 
a stochastic rainfall generator that simulate 6-hour, 
0.03° resolution rainfall fields over large-area river basins 
conditioned on global climate model data.

## Installation
`pip install stormlab-1.0-py3-none-any.whl`

## Dependencies
Required packages are listed in requirements.txt

## Usage
1. Download the example data and unzip at the code folder "/StormLab."
	- Google Drive: https://drive.google.com/file/d/1MJzO8bhKJxZ5sc7OuIQQQPw3AmQLKSME/view?usp=sharing
	- The folder should look like: "/StormLab/data" 
2. Rainstorm tracking
    - See `/examples/Storm_tracking_on_CESM2_data.ipynb`
3. TNGD distribution fitting
    - See `/examples/TNGD_distribution_fitting.ipynb`
4. Noise generation
    - See `/examples/Noise_generation.ipynb`
5. Rainfall simulation
    - See `/examples/Rainfall_simulation.ipynb`

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
* [David J. Lorenz](http://djlorenz.github.io/) - *research*

## Attribution
This project uses code from the following repositories:
- [pysteps](https://pysteps.readthedocs.io/en/stable/)
- [STREAM](https://github.com/sam-hartke/STREAM)
- [starch](https://github.com/lorenliu13/starch/tree/master)