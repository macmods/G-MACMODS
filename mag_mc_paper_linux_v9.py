#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:35:43 2021

@author: blsaenz
"""

import sys,os,copy,time
sys.path.append(r'/mnt/vat/data/mag/')
from magpy.mag_species import spp_p_dict,spp_p_dict_harvest
from multiprocessing import Pool
from magpy.mag_montecarlo import mc_run
from magpy.mag_util import param_update_no_overwrite

if os.name == 'nt':
    output_path = r'F:\mc_paper_v7'
else:
    output_path = r'/mnt/vat/data/mag/output/mc_paper_v9'


year=2017

mc_base_params = {

    'start_year': 2017,
    'run_name': "mc_paper_v9",
    'calc_steps': 365*2,
    'seeding_type': 1,
    'spinup_steps': 365,
    'fortran_calc': False,
    'monte_carlo_output': True,
    'repeat_annual_forcing': True,
    'output_path': output_path,
    'code_path': r'/mnt/vat/data/mag/magpy',
    'matlab_grid_filepath': r'/mnt/vat/data/CWM/grid/cwm_grid.mat',
    'default_cwm_mask': r'/mnt/vat/data/CWM/regions_and_masks/cwm_mask_20220412_from_hycom.h5',
    'suppress_calc_print': True,  # does nothing yet...
    'datetime_in_output': False,

}


forcing_var_meta = {
    'sst': {'freq': 8,
            'path': r'/mnt/vat/data/CWM/SST',
            'fname': r'sst_YyYy_gridinterp_patched.nc',
            'h5_path': 'sst',
            },
    'par': {'freq': 8,
            'path': r'/mnt/vat/data/CWM/PAR',
            'fname': r'par_YyYy_gridinterp.nc',
            'h5_path': 'par',
            },

    'chl': {'freq': 8,
            'path': r'/mnt/vat/data/CWM/chla',
            'fname': r'chl_YyYy_gridinterp.nc',
            'h5_path': 'chl',
            },
    'swh': {'freq': 1,
            'path': r'/mnt/vat/data/CWM/ECMWF',
            'fname': r'ecmwf_cwm_YyYy_9km_1day_gridinterp.nc',
            'h5_path': 'swh_mean',
            },
    'mwp': {'freq': 1,
            'path': r'/mnt/vat/data/CWM/ECMWF',
            'fname': r'ecmwf_cwm_YyYy_9km_1day_gridinterp.nc',
            'h5_path': 'mwp_mean',
            },
    'cmag': {'freq': 8,
             'path': r'/mnt/vat/data/CWM/hycom',
             'fname': r'HYCOM_YyYy_9km.nc',
             'h5_path': 'speed_mean',
             },
    'no3': {'freq': 5,
            'path': r'/mnt/vat/data/CWM/CESM/g.e11.G.T62_t12.eco',
            #'fname': r'cwm_CESM_8day_interp_NO3_20m.nc',
            'fname': r'cwm_CESM_5day_NO3_20m.nc',            
            'h5_path': 'NO3_20m',
            },
    'nflux': {'freq': 5,
              'path': r'/mnt/vat/data/CWM/CESM/g.e11.G.T62_t12.eco',
              'fname': r'cwm_CESM_5day_NO3_100m_Wflux.nc',
              'h5_path': 'NO3_100m_Wflux',
              },
    'seed': {'freq': 0,
             'path': r'/mnt/vat/data/mag/output/seed_paper_v9',
             'fname': r'Saccharina_seed_month_60pix.nc',
             'h5_path': 'seed_month',
             },
}

forcing_var_meta_local = {
    'sst': { 'freq': 8,
             'path': r'/home/blsaenz/data/CWM/SST',
             'fname': r'sst_YyYy_gridinterp_patched.nc',
             'h5_path': 'sst',
            },
    'par': { 'freq': 8,
             'path': r'/home/blsaenz/data/CWM/PAR',
             'fname': r'par_YyYy_gridinterp.nc',
             'h5_path': 'par',
             },
    'swh': { 'freq': 8,
             'path': r'/home/blsaenz/data/CWM/ECMWF',
             'fname': r'ecmwf_cwm_YyYy_9km_1day_gridinterp.nc',
             'h5_path': 'swh_mean',
             },
    'chl': { 'freq': 8,
             'path': r'/home/blsaenz/data/CWM/chla',
             'fname': r'chl_YyYy_gridinterp.nc',
             'h5_path': 'chl',
             },
    'mwp': { 'freq': 1,
            'path': r'/home/blsaenz/data/CWM/ECMWF',
            'fname': r'ecmwf_cwm_YyYy_9km_1day_gridinterp.nc',
            'h5_path': 'mwp_mean',
            },
    'cmag': { 'freq': 1,
             'path': r'/home/blsaenz/data/CWM/hycom',
             'fname': r'HYCOM_YyYy_9km.nc',
             'h5_path': 'speed_mean',
             },
    'no3': { 'freq': 5,
             'path': r'/home/blsaenz/data/CWM/2017_data_copy',
             'fname': r'cwm_CESM_5day_NO3_20m.nc',
             'h5_path': 'NO3_20m',
             },
    'nflux': { 'freq': 5,
             'path': r'/home/blsaenz/data/CWM/2017_data_copy',
             'fname': r'cwm_CESM_5day_NO3_100m_Wflux.nc',
             'h5_path': 'NO3_100m_Wflux',
             },
    'seed': { 'freq': 0,
             'path': r'/home/blsaenz/data/mag/output/seed_paper_v9',
             'fname': r'Saccharina_seed_month_60pix.nc',
             'h5_path': 'seed_month',
             },
}


# def fix_paths_for_windows(p,alt_win_path=None):
#     unix_path = r'/mnt/vat/'
#     if alt_win_path is None:
#         windows_path = r'X:/'
#     else:
#         windows_path =alt_win_path
#     for k,v in p.items():
#         if type(v) == str:
#             if unix_path in v:
#                 p[k] = v.replace(unix_path,windows_path)
#                 p[k] = os.path.normpath(p[k])
#         if type(v) == dict:
#             p[k] = fix_paths_for_windows(p[k],alt_win_path=r'C:/')
#     return p



def mc_run_iterations(niterations,run_name,params,forcing_var_meta,fortran):

    for ni in range(niterations):

        # will calculate 10 runs total, 2 paired runs for each spp
        for spp in ['Saccharina','Eucheuma','Sargassum','Porphyra','Macrocystis',]: # 
        #for spp in ['Porphyra','Macrocystis','Saccharina']:
        #for spp in ['Sargassum']:
            p = copy.deepcopy(params)
            p = param_update_no_overwrite(p,spp_p_dict_harvest[spp])
            #p = update_spp_harvest_params(spp,p)
            mc_run(spp,run_name,p,forcing_var_meta,fortran)


#if __name__ == '__main__':

niterations = 7 # multiply by 10 for actual runs, because we do each species[5]*flux_lim[2]


#forcing_var_meta = forcing_var_meta_local
nthreads = 36

# if os.name == 'nt':
#     mc_base_params = fix_paths_for_windows(mc_base_params)
#     forcing_var_meta = fix_paths_for_windows(forcing_var_meta)
#     nthreads = 24


pool = Pool(nthreads)

for i in range(nthreads):

    pool.apply_async(mc_run_iterations,args=(niterations,
                                "mc_paper_v9", # run_name stub
                                  mc_base_params,
                                  forcing_var_meta_local,
                                  #forcing_var_meta,
                                  False))
    time.sleep(1.5)
    # mc_run_iterations(niterations,
    #                               "mc_paper_v9", # run_name stub
    #                               mc_base_params,
    #                               forcing_var_meta_local,
    #                               #forcing_var_meta,
    #                               False)

pool.close()
pool.join()
del pool    
    
