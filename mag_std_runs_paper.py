#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:23:48 2021

@author: blsaenz
"""

import sys,os,time,copy
from calendar import isleap
sys.path.append(r'/mnt/vat/data/mag/')
from magpy import mag0
from multiprocessing import Pool
import numpy as np
import pandas as pd

from magpy.mag_util import generate_standard_runs,postprocess,get_mask,eez_mask,csv_from_pp,get_area

def std_prosprocess_opts(site_validation=False):
    #area = np.loadtxt(r'/mnt/vat/data/CWM/grid/area_twelfth_degree.txt')
    area = get_area(r'/mnt/vat/data/CWM/regions_and_masks/analysis_masks.mat','area')

    mask_dict = {'world':np.full((2160,4320),True,dtype=bool),
                 'shelf':get_mask(r'/mnt/vat/data/CWM/regions_and_masks/analysis_masks_20220412.nc','shelf_mask_100'),
                 'coast':get_mask(r'/mnt/vat/data/CWM/regions_and_masks/analysis_masks_20220412.nc','coast_9'),
                 'eez':eez_mask(r'/mnt/vat/data/mag/eez_masks.nc')}

    if site_validation:
        stats_dict = {v:['sum','mean','median'] for v in ['B',
                                                        'harv',]}
        netp = False
    else:
        stats_dict = {v:['sum','mean','median'] for v in ['Growth2',
                                                        'd_Be',
                                                        'd_Bm',
                                                        'd_Ns',
                                                        'harv',
                                                        'n_harv',
                                                        'B_N']}
        netp = True

    quantiles = ['all',0.9]
    quantile_var = 'harv'
    quantile_var_min = 150.0
    sums=True
    histograms=False
    sum_maps=True
    check=True
    mode='a',
    return (area,mask_dict,stats_dict,quantiles,quantile_var,
            quantile_var_min,sums,histograms,sum_maps,check,mode,netp)

def std_postprocess(spp,run_dir,sums=True,histograms=True,sum_maps=True,site_validation=False):
    (area,mask_dict,stats_dict,quantiles,quantile_var,
     quantile_var_min,_,_,_,check,mode,netp) = std_prosprocess_opts(site_validation)
    if spp=='Prophyra':
        quantile_var_min = 15.0
    postprocess([run_dir],area,mask_dict,stats_dict,quantiles,quantile_var,
                quantile_var_min,sums,histograms,sum_maps,check,mode,netp)


def std_postprocess_run_sets(run_sets,sums=True,histograms=True,sum_maps=True,site_validation=False):
    (area,mask_dict,stats_dict,quantiles,quantile_var,
     quantile_var_min,_,_,_,check,mode,netp) = std_prosprocess_opts(site_validation)
    for p,fvm in run_sets:
        run_dir = os.path.join(p['output_path'],p['run_name'])
        if p['spp'] == 'Prophyra':
            quantile_var_min = 15.0
        postprocess([run_dir],area,mask_dict,stats_dict,quantiles,quantile_var,
                    quantile_var_min,sums,histograms,sum_maps,check,mode,netp)

class gMACMODS_site(mag0.gMACMODS):

    def __init__(self, params, forcing_data):
        # do parent init()
        if not 'NO3_override' in params:
            raise ValueError("Expecting 'NO3_override' in params for gMACMODS_site simulation")
        super(gMACMODS_site, self).__init__(params, forcing_data)
        self.no3_f_ind = self.default_forcing_vars.index('no3')
        self.swh_f_ind = self.default_forcing_vars.index('swh')
        self.mwp_f_ind = self.default_forcing_vars.index('mwp')

    def external_pre_step_calc(self,nstep):
        if nstep==0: # and self.p['NO3_override'] > 0.0:
            # need to set initial Q correctly
            if self.p['seeding_type'] == self.init_seeding_type:
                no3_init = 35*0.85 #self.p['NO3_override']
                Q_init = self.p['mp_spp_Qmin'] + \
                    no3_init*(self.p['mp_spp_Qmax']-self.p['mp_spp_Qmin'])/35.0
                self.tracers['Q'][self.ocean_mask] = Q_init

        # overwrite forcing
        if self.p['NO3_override'] > 0.0:
            print('  overriding NO3 with',self.p['NO3_override'])
            self.f0[self.no3_f_ind][...] = self.p['NO3_override']
        if self.p['SWH_override'] > 0.0:
            print('  overriding SWH with',self.p['SWH_override'])
            self.f0[self.swh_f_ind][...] = self.p['SWH_override']
        if self.p['MWP_override'] > 0.0:
            print('  overriding MWP with',self.p['MWP_override'])
            self.f0[self.mwp_f_ind][...] = self.p['MWP_override']


common_forcing_var_meta = {
    'sst': { 'freq': 8,
             'path': r'/mnt/vat/data/CWM/SST',
             'fname': r'sst_YyYy_gridinterp_patched.nc',
             'h5_path': 'sst',
            },
    'par': { 'freq': 8,
             'path': r'/mnt/vat/data/CWM/PAR',
             'fname': r'par_YyYy_gridinterp.nc',
             'h5_path': 'par',
             },

    'chl': { 'freq': 8,
             'path': r'/mnt/vat/data/CWM/chla',
             'fname': r'chl_YyYy_gridinterp.nc',
             'h5_path': 'chl',
             },
    'swh': { 'freq': 1,
              'path': r'/mnt/vat/data/CWM/ECMWF',
              'fname': r'ecmwf_cwm_YyYy_9km_1day_gridinterp.nc',
              'h5_path': 'swh_mean',
              },
    'mwp': { 'freq': 1,
            'path': r'/mnt/vat/data/CWM/ECMWF',
            'fname': r'ecmwf_cwm_YyYy_9km_1day_gridinterp.nc',
            'h5_path': 'mwp_mean',
            },
    'cmag': { 'freq': 8,
             'path': r'/mnt/vat/data/CWM/hycom',
             'fname': r'HYCOM_YyYy_9km.nc',
             'h5_path': 'speed_mean',
             },
    'no3': { 'freq': 5,
                'path': r'/mnt/vat/data/CWM/CESM/g.e11.G.T62_t12.eco',
                'fname': r'cwm_CESM_5day_NO3_20m.nc',
                'h5_path': 'NO3_20m',
              },
    'nflux': { 'freq': 5,
             'path': r'/mnt/vat/data/CWM/CESM/g.e11.G.T62_t12.eco',
             'fname': r'cwm_CESM_5day_NO3_100m_Wflux.nc',
             #'fname': r'cwm_CESM_5day_NO3_HMXLmax_Wflux.nc',
             'h5_path': 'NO3_100m_Wflux',
             #'h5_path': 'NO3_HMXLmax_Wflux',
             },
    'seed': { 'freq': 0,
             'path': r'/mnt/vat/data/mag/output/seed_paper_v9',
             'fname': r'Saccharina_seed_month_60pix.nc',
             'h5_path': 'seed_month',
             },
}


def complete_run(model,p,fvm):
    p0 = mag0.build_run_params(p)
    model_instance = model(p0,fvm)
    model_instance.compute()
    del model_instance
    #std_postprocess(p0['spp'],
    #                os.path.join(p0['output_path'],p0['run_name']),
    #                sums=True,histograms=True,sum_maps=True)



def block_compute(run_sets,parallel=False,comp_threads=20,proc_threads=5,
                  model=mag0.gMACMODS,site_validation=False):

    ### Run mag0 in parallel block of comp_threads number of threads
    if parallel:
        i = 0
        nms = len(run_sets)
        while i < nms:
            pool = Pool(comp_threads)
            for j in range(comp_threads):
                if i < nms:
                    pool.apply_async(complete_run,args=(model,run_sets[i][0],run_sets[i][1]))
                    time.sleep(1.5)
                    i += 1
            pool.close()
            pool.join()
            del pool
    else:
        for p,fvm in run_sets:
            complete_run(model, p, fvm)

    ### Postprocess in separate loop - takes too much memory to have 20+ threads...
    if parallel:
        i = 0
        nms = len(run_sets)
        while i < nms:
            pool = Pool(proc_threads)
            for j in range(proc_threads):
                if i < nms:
                    p,fvm = run_sets[i]
                    p0 = mag0.build_run_params(p)
                    pool.apply_async(std_postprocess,args=(p0['spp'],
                            os.path.join(p0['output_path'],p0['run_name']),
                            True,False,True,site_validation))
                    time.sleep(5)
                    i += 1
            pool.close()
            pool.join()
            del pool
    else:
        for p,fvm in run_sets:
            p0 = mag0.build_run_params(p)
            std_postprocess(p0['spp'],
                            os.path.join(p0['output_path'],p0['run_name']),
                            sums=True,histograms=False,sum_maps=True,
                            site_validation=site_validation)



def std_runs_v9_lim_terms(parallel=True,mc_output=False,year=2017,
                          postprocess_only=False):

    output_path = r'/mnt/vat/data/mag/output/std/v9'
    seed_path = r'/mnt/vat/data/mag/output/seed_paper_v9'

    base_params = {

        'run_name': "std_lim_terms_v9_20230214_%i"%year,
        'start_year': year,
        'calc_steps': 365*2,
        'seeding_type': 1,
        'spinup_steps': 365,
        'fortran_calc': False,
        'monte_carlo_output': False,
        'repeat_annual_forcing': True,
        'output_path': output_path,
        'code_path': r'/mnt/vat/data/mag/magpy',
        'matlab_grid_filepath': r'/mnt/vat/data/CWM/grid/cwm_grid.mat',
        'default_cwm_mask': r'/mnt/vat/data/CWM/regions_and_masks/cwm_mask_20220412_from_hycom.h5',
        'suppress_calc_print': True,  # does nothing yet...
        'datetime_in_output': False,

        'B_freq':          8,
        'Q_freq':          8,
        'Gave_freq':       -1,
        'Dave_freq':       -1,
        'd_B_freq':       -1,
        'd_Q_freq':       -1,
        'Growth2_freq':   8,
        'd_Be_freq':      8,
        'd_Bm_freq':      8,
        'd_Ns_freq':      8,
        'harv_freq':      8,
        'GRate_freq':     8,
        'B_N_freq':       8,
        'n_harv_freq':    8,
        'min_lim_freq':    8,
        'gQ_freq':    8,
        'gT_freq':    8,
        'gE_freq':    8,
        'gH_freq':    8,}

    if mc_output:
        base_params['monte_carlo_output'] = True

    run_sets = generate_standard_runs(output_path,seed_path,base_params,common_forcing_var_meta)

    if postprocess_only:
        std_postprocess_run_sets(run_sets,sums=True,histograms=False,sum_maps=True)
    else:
        block_compute(run_sets,parallel,comp_threads=10)

    csv_from_pp([os.path.join(output_path,p['run_name']) for p,_ in run_sets],
        os.path.join(output_path,base_params['run_name']+'_sum_stats.csv'))



def std_runs_validation(parallel=True,year=2017):

    output_path = r'/mnt/vat/data/mag/output/std/breakage_output'
    seed_path = r'/mnt/vat/data/mag/output/seed_paper_v9'

    base_params = {

        'run_name': "validation_v8_paper_waves0_%i"%year,
        'start_year': year,
        'calc_steps': 365*2,
        'seeding_type': 1,
        'spinup_steps': 365,
        'fortran_calc': False,
        'monte_carlo_output': False,
        'repeat_annual_forcing': True,
        'output_path': output_path,
        'code_path': r'/mnt/vat/data/mag/magpy',
        'matlab_grid_filepath': r'/mnt/vat/data/CWM/grid/cwm_grid.mat',
        'default_cwm_mask': r'/mnt/vat/data/CWM/regions_and_masks/cwm_mask_20220412_from_hycom.h5',
        'suppress_calc_print': True,  # does nothing yet...
        'datetime_in_output': False,

        'B_freq':          8,
        'Q_freq':          8,
        'Gave_freq':       -1,
        'Dave_freq':       -1,
        'd_B_freq':       -1,
        'd_Q_freq':       -1,
        'Growth2_freq':   8,
        'd_Be_freq':      8,
        'd_Bm_freq':      8,
        'd_Ns_freq':      8,
        'harv_freq':      8,
        'GRate_freq':     8,
        'B_N_freq':       -1,
        'n_harv_freq':    8,
        'min_lim_freq':    8,
        'gQ_freq':    8,
        'gT_freq':    8,
        'gE_freq':    8,
        'gH_freq':    8,
    }

    run_sets = generate_standard_runs(output_path,seed_path,base_params,common_forcing_var_meta)

    for p,fvm in run_sets:
        #p['mp_spp_death'] = 0.003
        #p['mp_spp_E'] = 0.005
        p['SWH_override'] = 0.0
        p['MWP_override'] = 0.0
        p['NO3_override'] = -1


    rs_p = []
    for p,fvm in run_sets:
        #if p['spp']=='Eucheuma':
            if p['mp_N_flux_limit'] == 0:
                rs_p.append([p,fvm])
    run_sets = rs_p

    for p,fvm in run_sets:
        if p['spp']=='Saccharina' or p['spp']=='Macrocystis' or p['spp']=='Sargassum':
            # turn off harvest
            p['mp_harvest_schedule'] = 1
            p['mp_harvest_type'] = 1
            p['mp_harvest_freq'] = 365
            p['mp_harvest_span'] =  1
            p['run_name'] = p['run_name'] + '_no_harv'
            #p['mp_spp_kcap'] = kelp_kcap

        if p['spp']=='Porphyra':
            p['run_name'] = p['run_name'] + '_std_harv'

            p10 = copy.deepcopy(p)
            p10['run_name'] = p['run_name'] + '_no_harv'
            p10['mp_harvest_schedule'] = 1
            p10['mp_harvest_type'] = 1
            p10['mp_harvest_freq'] = 365
            p10['mp_harvest_span'] =  1


        elif p['spp']=='Eucheuma':
            p['mp_harvest_schedule'] = 0
            p['mp_harvest_type'] = 0
            p['mp_harvest_freq'] = 45
            p['mp_spp_kcap'] = 3000.
            p['mp_spp_seed'] =  400.
            p['run_name'] = p['run_name'] + '_fixedharv'

            # 200-g seed weigh runs
            p2 = copy.deepcopy(p)
            p2['mp_spp_seed'] =  200.
            p2['run_name'] = p2['run_name'] + '_seed200_kcap3.0'
            p2['mp_spp_kcap'] = 3000.

            p4 = copy.deepcopy(p)
            p4['mp_spp_seed'] =  600.
            p4['run_name'] = p4['run_name'] + '_seed600_kcap3.0'
            p4['mp_spp_kcap'] = 3000.

            p5 = copy.deepcopy(p)
            p5['mp_spp_seed'] =  400.
            p5['run_name'] = p5['run_name'] + '_seed400_kcap3.5'
            p5['mp_spp_kcap'] = 3500.

            p6 = copy.deepcopy(p)
            p6['mp_spp_seed'] =  200.
            p6['run_name'] = p6['run_name'] + '_seed200_kcap3.5'
            p6['mp_spp_kcap'] = 3500.

            p7 = copy.deepcopy(p)
            p7['mp_spp_seed'] =  600.
            p7['run_name'] = p7['run_name'] + '_seed600_kcap3.5'
            p7['mp_spp_kcap'] = 3500.

            p8 = copy.deepcopy(p)
            p8['mp_spp_seed'] =  400.
            p8['run_name'] = p8['run_name'] + '_seed400_kcap4.0'
            p8['mp_spp_kcap'] = 4000.

            p9 = copy.deepcopy(p)
            p9['mp_spp_seed'] =  200.
            p9['run_name'] = p9['run_name'] + '_seed200_kcap4.0'
            p9['mp_spp_kcap'] = 4000.

            p3 = copy.deepcopy(p)
            p3['mp_spp_seed'] =  600.
            p3['run_name'] = p3['run_name'] + '_seed600_kcap4.0'
            p3['mp_spp_kcap'] = 4000.


            p['run_name'] = p['run_name'] + '_seed400_kcap3.0'

    run_sets.append([p2,fvm])
    run_sets.append([p4,fvm])
    run_sets.append([p5,fvm])
    run_sets.append([p6,fvm])
    run_sets.append([p7,fvm])
    run_sets.append([p8,fvm])
    run_sets.append([p9,fvm])
    run_sets.append([p3,fvm])
    run_sets.append([p10,fvm])


    #std_postprocess_run_sets(run_sets,sums=True,histograms=True,sum_maps=True)

    block_compute(run_sets,parallel,comp_threads=20,model=gMACMODS_site)

    csv_from_pp([os.path.join(output_path,p['run_name']) for p,_ in run_sets],
        os.path.join(output_path,base_params['run_name']+'_sum_stats.csv'))



def std_runs_validation_sites(parallel=True,year=2017,forcing=common_forcing_var_meta):


    output_path = r'/mnt/vat/data/mag/output/std/validation_v9'
    seed_path = r'/mnt/vat/data/mag/output/seed_paper_v9'

    base_params = {

        'run_name': "validation_v9_Euch1750_%i"%year,
        'start_year': year,
        'calc_steps': 365,
        'seeding_type': 0,
        'spinup_steps': 0,
        'fortran_calc': False,
        'monte_carlo_output': False,
        'repeat_annual_forcing': True,
        'output_path': output_path,
        'code_path': r'/mnt/vat/data/mag/magpy',
        'matlab_grid_filepath': r'/mnt/vat/data/CWM/grid/cwm_grid.mat',
        'default_cwm_mask': r'/mnt/vat/data/CWM/regions_and_masks/cwm_mask_20220412_from_hycom.h5',
        'suppress_calc_print': True,  # does nothing yet...
        'datetime_in_output': False,

        # 'B_freq':          8,
        # 'Q_freq':          -1,
        # 'Gave_freq':       -1,
        # 'Dave_freq':       -1,
        # 'd_B_freq':       -1,
        # 'd_Q_freq':       -1,
        # 'Growth2_freq':   -1,
        # 'd_Be_freq':      -1,
        # 'd_Bm_freq':      -1,
        # 'd_Ns_freq':      -1,
        # 'harv_freq':      8,
        # 'GRate_freq':     -1,
        # 'B_N_freq':       -1,
        # 'n_harv_freq':    -1,
        # 'min_lim_freq':    -1,
        # 'gQ_freq':    -1,
        # 'gT_freq':    -1,
        # 'gE_freq':    -1,
        # 'gH_freq':    -1,

        'B_freq':          8,
        'Q_freq':          8,
        'Gave_freq':       -1,
        'Dave_freq':       -1,
        'd_B_freq':       -1,
        'd_Q_freq':       -1,
        'Growth2_freq':   8,
        'd_Be_freq':      8,
        'd_Bm_freq':      8,
        'd_Ns_freq':      8,
        'harv_freq':      8,
        'GRate_freq':     8,
        'B_N_freq':       -1,
        'n_harv_freq':    8,
        'min_lim_freq':    8,
        'gQ_freq':    8,
        'gT_freq':    8,
        'gE_freq':    8,
        'gH_freq':    8,
    }

    run_sets = generate_standard_runs(output_path,seed_path,base_params,common_forcing_var_meta)

    rs_p = []
    for p,fvm in run_sets:
        if p['spp']=='Eucheuma':
            if p['mp_N_flux_limit'] == 0:
                rs_p.append([p,fvm])
    run_sets = rs_p

    site_options = [
        [20, 45, 'CESM','Villanueva'],
        [150, 47, 'CESM', 'De_Goes_Reis'],
        [200, 45, 'CESM', 'Wang'],
        [200, 42, 'CESM', 'Msuya'],
        [25, 45, 'CESM', 'Ndobe'],
        [400, 45, 'CESM', 'Ganesan'],
        [200, 45, 'CESM', 'Hwang'],
        [200, 45, 'CESM', 'Philippines_paper'],
        [200, 45, 'CESM', 'Indonesia_paper'],
        [400, 45, 'CESM', 'Thirumaran'],
        [60, 42, 'CESM', 'Montoya_Rosas'],
        [100, 45, 1.5, 'Periyasami_2017'],
        [650, 45, 1.5, 'Periyasami_2019'],
        [100, 42, 2.8, 'Wakibia'],
        [30, 60, 13.5, 'Kotiya'],
        [400, 45, 17.5, 'Kumar'],
        [200, 42, 6.0, 'Wijayanto'],
        [400, 21, 'CESM', 'Nugroho'],
        [100, 45, 2.2, 'Periyasami_2017'],
        [650, 45, 3.5, 'Periyasami_2019'],
        [400, 45, 30.0, 'Kumar'],
        [100, 42, 3.3, 'Wakibia'],
        [30, 60, 1.3, 'Kotiya'],
        [250, 60, 1.3, 'Kotiya'],

        [400, 45, 30.0, 'Kumar'],
        [410, 45, 17.5, 'Kumar'],
        [410, 45, 30.0, 'Kumar'],
    ]

    run_sets_new = []
    for p,fvm in run_sets:
        if p['spp']=='Eucheuma':

            run_stub = p['run_name']

            p['mp_harvest_schedule'] = 0
            p['mp_harvest_type'] = 0
            p['mp_spp_kcap'] = 1750
            p['mp_spp_E'] = 0.01
            #p['mp_spp_death'] = 0.015

            p['SWH_override'] = 0.3
            p['MWP_override'] = 2.0

            for i,site in enumerate(site_options):
                p1 = p if i == 0 else copy.deepcopy(p)
                p1['mp_spp_seed'] = site[0]
                p1['mp_harvest_freq'] = site[1]
                if site[2] == 'CESM':
                    p1['NO3_override'] = -1
                    n_str = 'CESM'
                else:
                    p1['NO3_override'] = site[2]
                    n_str = '%3.1f'%site[2]
                p1['run_name'] = run_stub + '_seed-%i_harv-%i_NO3-%s_%s'%(site[0],site[1],n_str,site[3])
                if i > 0:
                    run_sets_new.append([p1, fvm])

    run_sets += run_sets_new

    block_compute(run_sets,parallel,comp_threads=18,model=gMACMODS_site,site_validation=True)

    csv_from_pp([os.path.join(output_path,p['run_name']) for p,_ in run_sets],
        os.path.join(output_path,base_params['run_name']+'_sum_stats.csv'))


def std_runs_validation_sites_temp_brown(parallel=True,year=2017,forcing=common_forcing_var_meta):


    output_path = r'/mnt/vat/data/mag/output/std/validation_v9'
    seed_path = r'/mnt/vat/data/mag/output/seed_paper_v9'

    base_params = {

        'run_name': "validation_v9_P_kc200_80_nowaves_%i"%year,
        'start_year': year,
        'calc_steps': 365,
        'seeding_type': 0,
        'spinup_steps': 0,
        'fortran_calc': False,
        'monte_carlo_output': False,
        'repeat_annual_forcing': True,
        'output_path': output_path,
        'code_path': r'/mnt/vat/data/mag/magpy',
        'matlab_grid_filepath': r'/mnt/vat/data/CWM/grid/cwm_grid.mat',
        'default_cwm_mask': r'/mnt/vat/data/CWM/regions_and_masks/cwm_mask_20220412_from_hycom.h5',
        'suppress_calc_print': True,  # does nothing yet...
        'datetime_in_output': False,

        'B_freq':          8,
        'Q_freq':          -1,
        'Gave_freq':       -1,
        'Dave_freq':       -1,
        'd_B_freq':       -1,
        'd_Q_freq':       -1,
        'Growth2_freq':   -1,
        'd_Be_freq':      -1,
        'd_Bm_freq':      -1,
        'd_Ns_freq':      -1,
        'harv_freq':      8,
        'GRate_freq':     -1,
        'B_N_freq':       -1,
        'n_harv_freq':    0,
        'min_lim_freq':    -1,
        'gQ_freq':    -1,
        'gT_freq':    -1,
        'gE_freq':    -1,
        'gH_freq':    -1,
    }

    run_sets = generate_standard_runs(output_path,seed_path,base_params,common_forcing_var_meta)

    rs_p = []
    for p,fvm in run_sets:
        if p['spp']=='Porphyra': #p['spp']=='Sargassum' or : #p['spp']=='Saccharina' or p['spp']=='Macrocystis': # or  :
            if p['mp_N_flux_limit'] == 0:
                rs_p.append([p,fvm])
    run_sets = rs_p


    site_options = [
        # [init weight, harvest days, no3, name, waves, init_day]
        [50, 151,    18.0,   'Augyte','ECMWF',12],
        [50, 227+15, 'CESM', 'Bak_et_al','ECMWF',12],
        [50, 243,    8.0,    'Boderskov','ECMWF',10],
        [50, 61,     5.0,    'Peteiro_2014','ECMWF',4],
        [50, 90,     7.0,    'Peteiro_2013','ECMWF',1],
        [50, 184,    'CESM', 'Gutierrez','ECMWF',5],
        [50, 184,    'CESM', 'Correa_May','ECMWF',5],
        [50, 92,     'CESM', 'Correa_Mar','ECMWF',3],
        [50, 185,    'CESM', 'Park','ECMWF',1],
        [50, 335,    3.0,    'Marinho','ECMWF',6],
        [50, 184,    'CESM', 'Biancacci','ECMWF',5],
        [50, 151,    'CESM', 'Hargrave',0,12],
        [50, 365,    8.0,    'Camus','ECMWF',-1],
        [50, 121,    'CESM', 'Freitas','ECMWF',12],

        [50, 365,    'CESM', 'no_harvest','ECMWF',-1],
    ]


    site_options = [
        # [init weight, harvest days, no3, name, waves, init_day]
        [50, 184,     'CESM', 'Forbord_v2','ECMWF',3],
        [50, 303,     'CESM', 'Camus_v2','ECMWF',10],
        [50, 335,     'CESM', 'Marinho_v2','ECMWF',6],
        [50, 120,     'CESM', 'Macchiavello_v2','ECMWF',9],

    ]

    site_options_m1mo = [
        # [init weight, harvest days, no3, name, waves, init_day]
        [50, 131,    18.0,   'Augyte','ECMWF',1],
        [50, 211, 'CESM', 'Bak_et_al','ECMWF',1],
        [50, 213,    8.0,    'Boderskov','ECMWF',11],
        [50, 31,     5.0,    'Peteiro_2014','ECMWF',5],
        [50, 60,     7.0,    'Peteiro_2013','ECMWF',2],
        [50, 154,    'CESM', 'Gutierrez','ECMWF',6],
        [50, 154,    'CESM', 'Correa_May','ECMWF',6],
        [50, 62,     'CESM', 'Correa_Mar','ECMWF',4],
        [50, 155,    'CESM', 'Park','ECMWF',2],
        [50, 305,    3.0,    'Marinho','ECMWF',7],
        [50, 154,    'CESM', 'Biancacci','ECMWF',6],
        [50, 131,    'CESM', 'Hargrave',0,1],
        [50, 365,    8.0,    'Camus','ECMWF',-1],
        [50, 91,    'CESM', 'Freitas','ECMWF',1],

        [50, 365,    'CESM', 'no_harvest','ECMWF',-1],
    ]

    #site_options_m1mo = [
    #    [50, 184,    2.75, 'Biancacci','ECMWF',5],
    #    [50, 154,    2.75, 'Biancacci_m1mo','ECMWF',6],
    #]

    #site_options = [
    #    [50, 92,     'CESM', 'Correa_Mar','ECMWF',3],
    #]


    #     [50, -1, 'CESM', 'Macchiavello','ECMWF',-1],
    #     [50, -1, 'CESM', 'Forbord','ECMWF',-1],
    #     [50, -1, 'CESM', 'Gundersen','ECMWF',-1],
    #     [50, -1, 'CESM', 'Krumhansl','ECMWF',-1],
    #     [50, -1, 'CESM', 'van_Son','ECMWF',-1],
    #     [50, -1, 'CESM', 'Ulaski','ECMWF',-1],
    #     [50, -1, 'CESM', 'Pesarrodona','ECMWF',-1],
    #     [50, -1, 'CESM', 'Smale','ECMWF',-1],
    #     [50, -1, 'CESM', 'Pedersen','ECMWF',-1],
    # ]

    site_options_porphyra = [
        # [init weight, harvest days, no3, name, waves, init_day]
        [10, 121,    20.0,   'Wu','ECMWF',12],
        [10, 121,    'CESM', 'Wang','ECMWF',12],
        [10, 181,    39.0,    'He','ECMWF',10],
        #[50, 121,    'CESM',      'Hwang','ECMWF',12],
        [10, 365,     'CESM',      'no_harvest','ECMWF',-1],
    ]

    site_options_porphyra_15d = [
        # [init weight, harvest days, no3, name, waves, init_day]
        [10, 115,    20.0,   'Wu',0,12],
        [10, 115,    'CESM', 'Wang',0,12],
        [10, 175,    39.0,    'He',0,10],
        #[50, 121,    'CESM',      'Hwang','ECMWF',12],
        [10, 365,     'CESM',      'no_harvest',0,-1],
    ]


    site_options_porphyra_m1mo = [
        # [init weight, harvest days, no3, name, waves, init_day]
        [10, 91,    20.0,   'Wu','ECMWF',1],
        [10, 91,    'CESM', 'Wang','ECMWF',1],
        [10, 151,    39.0,    'He','ECMWF',11],
        [10, 91,    'CESM',      'Wang','ECMWF',1],
        [10, 365,     'CESM',      'no_harvest','ECMWF',-1],
    ]


    site_options_sargassum = [
        # [init weight, harvest days, no3, name, waves, init_day]
        [50, 365,    'CESM',   'sargassum_waves','ECMWF',-1],
        [50, 365,   'CESM', 'sargassum_no-waves',0,-1],
        [50, 365,    9.5,    'sargassum_N9.5', 'ECMWF',-1],
    ]


    run_sets_new = []
    for p,fvm in run_sets:
            run_stub = p['run_name']

            # setup some harvesting
            if p['spp']=='Saccharina' or p['spp']=='Macrocystis':

                p['mp_harvest_schedule'] = 1
                p['mp_harvest_type'] = 0
                p['mp_harvest_span'] =  0
                p['mp_harvest_nmax'] = 1
                #p['mp_spp_kcap'] = 1600.
                p['mp_spp_E'] = 0.01
                #p['mp_spp_death'] = 0.015

                so = site_options
                #so = site_options_m1mo

            elif p['spp']=='Porphyra':

                p['mp_harvest_schedule'] = 2
                p['mp_harvest_type'] = 3  # 1 = fractional instead of seed weight
                p['mp_spp_kcap'] = 200.0 # p['mp_spp_kcap']*0.54
                p['mp_spp_E'] = 0.01
                #p['mp_spp_death'] = 0.015

                so = site_options_porphyra_15d
                #so = site_options_porphyra_m1mo

            elif p['spp']=='Sargassum':

                p['mp_harvest_schedule'] = 1
                p['mp_harvest_type'] = 0
                p['mp_harvest_span'] = 0
                p['mp_harvest_nmax'] = 1
                p['mp_spp_kcap'] = 500.0 #p['mp_spp_kcap']*0.55
                p['mp_spp_E'] = 0.01
                #p['mp_spp_death'] = 0.015

                so = site_options_sargassum

            # setup seeding and nutrients and waves and harvest dates, if any
            for i,site in enumerate(so):
                p1 = p if i == 0 else copy.deepcopy(p)

                p1['mp_spp_seed'] = site[0]
                p1['mp_harvest_freq'] = site[1]
                if site[2] == 'CESM':
                    p1['NO3_override'] = -1
                    n_str = 'CESM'
                else:
                    p1['NO3_override'] = site[2]
                    n_str = '%3.1f'%site[2]

                if site[4] == 'ECMWF':
                    p1['SWH_override'] = -1
                    p1['MWP_override'] = -1
                else:
                    p1['SWH_override'] = 0.3
                    p1['MWP_override'] = 2.0

                if site[5] == -1:

                    # for porphyra, need to reset this
                    if p['spp']=='Porphyra':
                        p1['mp_harvest_schedule'] = 1
                        p1['mp_harvest_type'] = 1
                        p1['mp_harvest_span'] =  0
                        p1['mp_harvest_nmax'] = 6

                    p1['seeding_type'] = 1
                    p1['calc_steps'] = 365*2
                    p1['spinup_steps'] = 365
                    p1['run_name'] = run_stub + '_harv-%i_NO3-%s_seedtype-1_%s'%(site[1],n_str,site[3])
                else:
                    # for porphyra, need to reset this
                    if p1['spp']=='Porphyra':
                        p1['mp_harvest_span'] =  site[1] - 40
                        p1['mp_harvest_nmax'] = int(p1['mp_harvest_span']/15)
                        if p1['mp_harvest_type'] == 3:
                            p1['mp_harvest_kg'] = 0.08


                    p1['seeding_type'] = 0
                    p1['calc_steps'] = 365
                    p1['spinup_steps'] = 0
                    p1['start_month'] = site[5]
                    p1['start_day'] = 15  # start on 15th of month always
                    p1['run_name'] = run_stub + '_harv-%i_NO3-%s_seedmo-%i_%s'%(site[1],n_str,p1['start_month'],site[3])
                if i > 0:
                    run_sets_new.append([p1, fvm])


    run_sets += run_sets_new

    block_compute(run_sets,parallel,comp_threads=36,model=gMACMODS_site,site_validation=True)

    csv_from_pp([os.path.join(output_path,p['run_name']) for p,_ in run_sets],
        os.path.join(output_path,base_params['run_name']+'_sum_stats.csv'))



def std_runs_multiyear(parallel=True,forcing=common_forcing_var_meta):

    output_path = r'/mnt/vat/data/mag/output/std/multiyear_v9'
    seed_path = r'/mnt/vat/data/mag/output/seed_paper_v9'

    base_params = {

        'run_name': "std_v9_multiyear",
        'start_year': 2003,
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

        'B_freq':          -1,
        'Q_freq':          -1,
        'Gave_freq':       -1,
        'Dave_freq':       -1,
        'd_B_freq':       -1,
        'd_Q_freq':       -1,
        'Growth2_freq':   0,
        'd_Be_freq':      0,
        'd_Bm_freq':      0,
        'd_Ns_freq':      0,
        'harv_freq':      0,
        'GRate_freq':     -1,
        'B_N_freq':       -1,
        'n_harv_freq':    0,
        'min_lim_freq':    -1,

        'gQ_freq':    -1,
        'gT_freq':    -1,
        'gE_freq':    -1,
        'gH_freq':    -1,
    }

    # switchng back to 8-day waves, since we dont have it processed into 1-day
    forcing_multiyear = copy.deepcopy(forcing)
    forcing_multiyear['swh'] = { 'freq': 8,
              'path': r'/mnt/vat/data/CWM/ECMWF',
              'fname': r'ecmwf_cwm_YyYy_9km_gridinterp.nc',
              'h5_path': 'swh_mean',
              }
    forcing_multiyear['mwp'] = { 'freq': 8,
            'path': r'/mnt/vat/data/CWM/ECMWF',
            'fname': r'ecmwf_cwm_YyYy_9km_gridinterp.nc',
            'h5_path': 'mwp_mean',
            }

    run_sets = generate_standard_runs(output_path,seed_path,base_params,forcing_multiyear)

    #rs_p = []
    #for p,fvm in run_sets:
    #    #if p['spp']=='Porphyra':
    #    if p['mp_N_flux_limit']==1:
    #        rs_p.append([p,fvm])
    #run_sets = rs_p

    multiyear_sets = []

    for y in range(2003,2020):
        r1 = copy.deepcopy(run_sets)
        for i in range(len(run_sets)):
            r1[i][0]['start_year'] = y
            r1[i][0]['run_name'] = "%i_"%y + r1[i][0]['run_name']
            if isleap(y):
                r1[i][0]['calc_steps'] += 1
        multiyear_sets += r1

    #for p,fvm in multiyear_sets:
    #    std_postprocess_run_sets([[p,fvm]])


    block_compute(multiyear_sets,parallel,comp_threads=20)

    csv_from_pp([os.path.join(output_path,p['run_name']) for p,_ in multiyear_sets],
        os.path.join(output_path,base_params['run_name']+'_sum_stats.csv'))



std_runs_v9_lim_terms(parallel=True,mc_output=False,year=2017,unoptimize_harvest=False)
#std_runs_multiyear(parallel=True,forcing=common_forcing_var_meta)
#std_runs_validation(parallel=True,year=2017)
#std_runs_validation_sites_temp_brown(parallel=True,year=2017)
#std_runs_validation_sites(parallel=True,year=2017)







