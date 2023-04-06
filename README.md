# G-MACMODS
Global MacroAlgae Cultivation MODeling System (G-MACMODS)
============================================================================
Isabella Arzeno-Soltero
Benjamin T. Saenz
Kristen Davis

Updated: 2023-04-04

The full G-MACMODS code can be found at [ https://github.com/macmods/G-MACMODS.git].

G-MACMODS (Arzeno-Soltero et al. 2023) is a derivative of MACMODS, utilizing
the same conceptual framework, and in many cases parameterizations, as the earlier model described in Frieder et al.,2022. Briefly,
the main model differences include using 0D (single set of tracers) for the
water column, seaweed type-specific parameterizations for up to 4 different seaweed
types [2 temperate browns ('Saccharina','Macrocystis'), temperate red
('Pyropia/Porphyra'), tropical brown ('Sargassum'), and tropical red
('Eucheuma')], and a crowding parameterization derived from empirical growth
data (Xiao et al., 2019) and type-specific tuning. See Arzeno-Soltero et al.
2023, including supplementary information, for a full model description.

G-MACMODS was translated into Python-3 for the purposes of generating and
analyzing large numbers of simulations in a Monte Carlo manner, because at
the time of model creation, seaweed biophysical parameters were very
uncertain and a gross estimate of model error were needed.  Python and
various accelerated numerical and analysis packages provided an efficient
scalability through parallelization.

The model code package 'magpy' contains model classes to generate a single
0D simulation (MAG0), and a G-MACMODS simulation, with its global
input data dependencies(Arzeno-Soltero et al. 2023).Driver python scripts which
created the simulations used in Arzeno-Soltero et al. 2023 are found in
the outer directory that generate 'standard' simulations, simulations
used for validation comparisons against a suite of other publications,
and Monte Carlo simulations for estimating uncertainty.


Usage Guidelines and Notes:
---------------------------

-- The seaweed type parameterizations from Arzeno-Soltero et al. 2023
(see mag_species.py) were developed from a combination of values derived
from literature, and where unavailable, from tuning, such that model output
was within the range of wild and farmed seaweed biomass and growth
rates that were observed *in the oceans* (not in a lab setting). Modification
of these parameters will result in simulations not supported by research.
Use of new seaweed type parameterizations with G-MACMODS should always be
accompanied by supporting validation simulations and experimental data.

-- The seaweed genus named mentioned above were the primary sources for
parameters for modeled seaweed types, however the types modeled here should
not be seen as representative for any single species. Typically, G-MACMODS
seaweed types have wider tolerances for environment that most single species,
as our attempt was to model yield potential, and one of our assumptions is
that strain selection will be used to select cultivars that are suited to
the local aquaculture environment.


Environmental inputs used in Arzeno-Soltero et al. (2023)
---------------------------

Surface nitrate concentrations and vertical nitrate fluxes:
Long, M., B. Saenz. (2023). Nitrate flux and inventory from high-resolution CESM CORE-Normal-Year integration. Version 1.0. UCAR/NCAR - GDEX. https://doi.org/10.5065/hpae-3j62.

MODIS sea surface temperature (SST), MODIS surface photosynthetically active radiation (PAR), and net oceanic primary productivity (NPP) were downloaded from the Ocean Productivity website (https://sites.science.oregonstate.edu/ocean.productivity/index.php). Specifically, 8-day NPP can be found at http://orca.science.oregonstate.edu/1080.by.2160.8day.hdf.vgpm.m.chl.m.sst.php, whereas 8-day MODIS inputs can be found at https://sites.science.oregonstate.edu/ocean.productivity/1080.by.2160.8day.inputData.php.

Zonal and meridional surface current velocities were taken from the HYbrid-Coordinate Ocean Model (HYCOM99) Global Ocean Forecasting System (GOFS) 3.1, accessed from https://www.hycom.org/dataserver/gofs-3pt1/analysis.

Significant wave height and wave period were taken from the European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 atmospheric reanalysis:
Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N.: ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d4.
---------------------------

