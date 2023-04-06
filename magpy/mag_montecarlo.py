import sys,random,string,copy
sys.path.append(r'/mnt/reservior/data/mag/')
from magpy import mag0
from magpy.mag_util import seed_paper_name_stub
import numpy as np

# what to save - full grid:
# -------------------------
# total biomass growth
# total death
# total exudation
# total harvested
# total nutrient uptake
# hopefully death+harvested=total growth

random.seed()

# 'Eucheuma','Sargassum','Saccharina','Macrocystis','Porphyra'

mc_p_bounds = {
    
    'Eucheuma': {
        'mp_spp_Vmax': [4.05*24, 16.25*24],           # [umol N/g-DW/d], excpecting umol N/m2/d, conversion below - All spp
        'mp_spp_Ks_NO3': [0.15*1000, 13.75*1000],       # [umol N/m3]  % L. Roberson - All spp
        'mp_spp_kcap': [1500., 2500.],             # [g(dry)/m]
        'mp_spp_Gmax_cap': [0.1, 0.3],             # [1/day]  - All spp
        'mp_spp_PARs': [52.05/4.57, 550/4.57],       # [W/m2]   - All spp
        'mp_spp_PARc': [3.75/4.57, 32.5/4.57],       # [W/m2]
        'mp_spp_Qmin': [4.32, 7.2],            # [mg N/g(dry)]
        'mp_spp_Qmax': [33, 55.],           # [mg N/g(dry)]
        'mp_spp_CD': [0.01, 1.],             # drag coefficient (unitless)
        'mp_spp_dry_sa': [71.1, 118.5],  # [m2/g(wet)]
        'mp_spp_E': [0.0005,0.0195],            # [d-1] % of
        'mp_spp_death': [0.003, 0.017],         # death rate [1/day]
        'mp_wave_mort_factor': [0.3, 1.7]         # wave mortality scaler [unitless, but multiplied against 1/day]
        },

    'Sargassum': {
        'mp_spp_Vmax': [1.86*24, 36.9*24],           # [umol N/g-DW/d], excpecting umol N/m2/d, conversion below - All spp
        'mp_spp_Ks_NO3': [1.12*1000, 5.5*1000],       # [umol N/m3]  % L. Roberson - All spp
        'mp_spp_kcap': [375., 625.],             # [g(dry)/m]
        'mp_spp_Gmax_cap': [0.1, 0.3],             # [1/day]  - All spp
        'mp_spp_PARs': [112.5/4.57, 643.75/4.57],       # [W/m2]   - All spp
        'mp_spp_PARc': [3.75/4.57, 46.25/4.57],       # [W/m2]
        'mp_spp_Qmin': [4.32, 7.2],            # [mg N/g(dry)]
        'mp_spp_Qmax': [33, 55.],           # [mg N/g(dry)]
        'mp_spp_CD': [0.01, 1.],             # drag coefficient (unitless)
        'mp_spp_dry_sa': [249.75, 416.25],  # [m2/g(wet)]
        'mp_spp_E': [0.0005,0.0195],            # [d-1] % of
        'mp_spp_death': [0.003, 0.017],         # death rate [1/day]
        'mp_wave_mort_factor': [0.3, 1.7]         # wave mortality scaler [unitless, but multiplied against 1/day]
        },

    'Saccharina': {
        'mp_spp_Vmax': [1.9*24, 30.0*24],           # [umol N/g-DW/d], excpecting umol N/m2/d, conversion below - All spp
        'mp_spp_Ks_NO3': [1.05*1000, 4.2*1000],       # [umol N/m3]  % L. Roberson - All spp
        'mp_spp_kcap': [1500., 2500.],             # [g(dry)/m]
        'mp_spp_Gmax_cap': [0.1, 0.3],             # [1/day]  - All spp
        'mp_spp_PARs': [11.25/4.57, 212.5/4.57],       # [W/m2]   - All spp
        'mp_spp_PARc': [5.7/4.57, 29.25/4.57],       # [W/m2]
        'mp_spp_Qmin': [7.635, 12.725],            # [mg N/g(dry)]
        'mp_spp_Qmax': [40.5, 67.5],           # [mg N/g(dry)]
        'mp_spp_CD': [0.01, 1.],             # drag coefficient (unitless)
        'mp_spp_dry_sa': [43.5, 72.5],  # [m2/g(wet)]
        'mp_spp_E': [0.0005,0.0195],            # [d-1] % of
        'mp_spp_death': [0.003, 0.017],         # death rate [1/day]
        'mp_wave_mort_factor': [0.3, 1.7]         # wave mortality scaler [unitless, but multiplied against 1/day]
        },

    'Macrocystis': {
        'mp_spp_Vmax': [2.25*24, 38.125*24],           # [umol N/g-DW/d], excpecting umol N/m2/d, conversion below - All spp
        'mp_spp_Ks_NO3': [3.2*1000, 18.1*1000],       # [umol N/m3]  % L. Roberson - All spp
        'mp_spp_kcap': [1500., 2500.],             # [g(dry)/m]
        'mp_spp_Gmax_cap': [0.1, 0.3],             # [1/day]  - All spp
        'mp_spp_PARs': [105.75/4.57, 325./4.57],       # [W/m2]   - All spp
        'mp_spp_PARc': [7.5/4.57, 43.125/4.57],       # [W/m2]
        'mp_spp_Qmin': [7.635, 12.725],            # [mg N/g(dry)]
        'mp_spp_Qmax': [40.5, 67.5],           # [mg N/g(dry)]
        'mp_spp_CD': [0.01, 1.],             # drag coefficient (unitless)
        'mp_spp_dry_sa': [43.5, 72.5],  # [m2/g(wet)]
        'mp_spp_E': [0.0005,0.0195],            # [d-1] % of
        'mp_spp_death': [0.003, 0.017],         # death rate [1/day]
        'mp_wave_mort_factor': [0.3, 1.7]         # wave mortality scaler [unitless, but multiplied against 1/day]
        },

    'Porphyra': {
        'mp_spp_Vmax': [26.25*24, 90*24],           # [umol N/g-DW/d], excpecting umol N/m2/d, conversion below - All spp
        'mp_spp_Ks_NO3': [1.5*1000, 12.7*1000],       # [umol N/m3]  % L. Roberson - All spp
        'mp_spp_kcap': [90., 160.],             # [g(dry)/m]
        'mp_spp_Gmax_cap': [0.1, 0.3],             # [1/day]  - All spp
        'mp_spp_PARs': [34.5/4.57, 233.75/4.57],       # [W/m2]   - All spp
        'mp_spp_PARc': [6.75/4.57, 54./4.57],       # [W/m2]
        'mp_spp_Qmin': [7.635, 12.725],            # [mg N/g(dry)]
        'mp_spp_Qmax': [40.5, 67.5],           # [mg N/g(dry)]
        'mp_spp_CD': [0.01, 1.],             # drag coefficient (unitless)
        'mp_spp_dry_sa': [7.5, 12.5],  # [m2/g(wet)]
        'mp_spp_E': [0.0005,0.0195],            # [d-1] % of
        'mp_spp_death': [0.003, 0.017],         # death rate [1/day]
        'mp_wave_mort_factor': [0.3, 1.7]         # wave mortality scaler [unitless, but multiplied against 1/day]
        },

}


def random_run_name(name,length=12):
    letters = string.ascii_letters + string.digits
    return name+'_' + ''.join(random.choice(letters) for i in range(length))


def randomize_params(spp,params,run_name,randomize_harvest_freq=False,
                     randomize_harvest_f=False,randomize_start_year=False,
                     custom_randomization=[]): #,dist='uniform'):

    my_params = copy.deepcopy(params)

    my_params['run_name'] = random_run_name(run_name,length=12)

    if not custom_randomization:
        random_vars = mc_p_bounds[spp]
    else:
        random_vars = {k:mc_p_bounds[spp][k] for k in custom_randomization}

    for param,bounds in random_vars.items():
        my_params[param] = random.uniform(bounds[0],bounds[1])
    if 'mp_spp_Vmax' in random_vars.keys():
        my_params['mp_spp_Vmax'] = my_params['mp_spp_Vmax'] * my_params['mp_spp_dry_sa']  # should just do this in the model...

    if randomize_harvest_freq:
        harvest_freq = [45, 60, 90, 120, 180]
        my_params['mp_harvest_freq'] = harvest_freq[random.randint(0,4)]

    if randomize_harvest_f:
        harvest_f = [0.4, 0.6, 0.8,]
        my_params['mp_harvest_f'] = harvest_f[random.randint(0,2)]

    if randomize_start_year:
        year = np.arange(2003,2020)
        my_params['start_year'] = year[random.randint(0,len(year)-1)]

    return my_params


def mc_run(spp,run_name,params,forcing_var_meta,
           randomize_harvest_freq=False,fortran=False,seed_ext='_modef9',flux_lim=True,
           custom_randomization=[]):

    # seed_fname needs to be in params for mc runs
    seed_name_stub = seed_paper_name_stub(params)
        
    # randomize params
    my_params = randomize_params(spp,params,run_name,
                                 randomize_harvest_freq=randomize_harvest_freq,
                                 custom_randomization=custom_randomization)    
    my_params['spp'] = spp  # redundant?
    my_params['fortran_calc'] = fortran


    # Run both flux_lim and ambient
    rn = my_params['run_name']
    fopt = [0,1] if flux_lim else [0]
    for flux_lim in fopt:
        # update output and run_name so that flux_limit and ambient are in one folder?
        my_params['mp_N_flux_limit'] = flux_lim
        my_params['run_name'] = rn + '_f%i'%flux_lim

        if params['seeding_type'] != 0:
            #forcing_var_meta['seed']['fname'] = seed_name_stub + '_f%i'%flux_lim + '_seed_month_60pix.nc'
            forcing_var_meta['seed']['fname'] = seed_name_stub + '_f0' + '_seed_month_multiyear' + seed_ext + '.nc'

        p = mag0.build_run_params(my_params)
        model = mag0.gMACMODS(p,forcing_var_meta=forcing_var_meta)
        model.compute()
        

