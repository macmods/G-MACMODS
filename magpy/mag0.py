'''
Released with G-MACMODS.  Main classes for building and computing the a
MAG0-derived macroaglal grwoth simulations

Kristen Davis
Christina Fiedler
Isabella Arenzo-Soltero
Benjamin Saenz


'''
import os,sys,shutil,csv,pickle,math,time,datetime
from calendar import isleap
import numpy as np
import h5py
from numba import jit
#from numpy import asfortranarray,ascontiguousarray
from contextlib import contextmanager
import netCDF4
from netCDF4 import date2num # use these with date2num(dt,'days since %i-01-01'%year) to convert!
import pandas as pd
import xarray as xr

try:
    from .mag_calc_fortran import mag_calc as mag_calc_f
except:
    print('mag_calc_fortran not imported/available!')



os.environ['OMP_NUM_THREADS'] = '10'

default_cwm_mask = r'X:/data/CWM/scripts/python/mask_cbpm_2021_01_13.txt'
default_cwm_grid_area = r'X:\data\CWM\grid\area_twelfth_degree.txt'

# example to write hdf5 mask from txt mask:
# mask=np.loadtxt(r'/mnt/reservior/data/CWM/scripts/python/mask_cbpm_2021_01_13.txt')
# mask=mask.astype(np.int32)
# h5f = h5py.File(r'/mnt/reservior/data/CWM/scripts/python/mask_cbpm_2021_01_13.h5', 'w')
# h5f.create_dataset('cwm_ocean_mask',data=mask)
# h5f.close()

def load_cwm_grid(mask_fn=default_cwm_mask):
    longitude = np.arange(4320,dtype=np.float32) * 1/12 - 180 + 1/24
    latitude = -1.0*(np.arange(2160,dtype=np.float32) * 1/12 - 90 + 1/24)
    xlon, ylat = np.meshgrid(longitude, latitude) # output coordinate meshes
    if mask_fn.endswith('h5'):
        h5f = h5py.File(mask_fn,'r')
        mask = h5f['cwm_ocean_mask'][:]
        h5f.close()
    else:
        mask = np.loadtxt(mask_fn)
    mask = np.logical_not(mask)
    return longitude,latitude,xlon,ylat,mask

def load_cwm_grid_area_bounds(mask_fn=default_cwm_mask):
    longitude,latitude,xlon,ylat,mask = load_cwm_grid(mask_fn)
    lonb = np.arange(4321) * 1/12 - 180
    latb = -1.0*(np.arange(2161) * 1/12 - 90)
    xlonb,ylatb = np.meshgrid(lonb, latb)
    area = np.loadtxt(default_cwm_grid_area)
    return longitude,latitude,xlon,ylat,mask,area,lonb,latb,xlonb,ylatb

def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def find_nearest(array, value):
    return array[find_nearest_idx(array,value)]

def step_event_from_freq(model_step_var,freq,max_steps=365,max_count=None):
    """
    Create an array, with one value for each model step, that indicates if something
    periodic (as specific by freq) happens on the model step. Returns and array
    or len(model_step_var) with values = -1 if no event, and an ascending integer
    for events, up to max_count.  I'm not sure the different any more between
    max_steps and max_count ... need simplifying/help

    Parameters
    ----------
    model_step_var : numpy ndarray, with the same number of elements as model steps
        for now this seems to need to start with 0, maybe remove this in future
        this function does not realy worth with this as a "variable".
    freq : integer describing the model step freqeuncy for when things happen
        DESCRIPTION.
    max_steps : TYPE, optional
        DESCRIPTION. The default is 365.
    max_count : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    event_check : see description

    """
    if freq == 0:
        # only write out at end of run
        event_check = np.full(len(model_step_var),-1,np.int32)
        event_check[-1] = 0

    else:
        if max_count is None:
            max_count = 1000000 # something big

        # map freq to model steps
        steps_as_freq = np.arange(0,max_steps,freq)

        # find start in freq space
        next_event = find_nearest_idx(steps_as_freq,model_step_var[0])
        if steps_as_freq[next_event] > model_step_var[0]:
            next_event -= 1

        # store actual time slice number, at first change of doy
        event_check = np.full(len(model_step_var),-1,np.int32)  # -1 means don't do anything
        for i,step in enumerate(model_step_var):
            if step == steps_as_freq[next_event] and next_event <= max_count-1:  #
                event_check[i] = next_event
                next_event += 1
                if next_event >= len(steps_as_freq):
                    next_event = 0 # cycle back around, b/c forcing files are only 1 yar long
    return event_check


def create_gMACMODS_output_dataset_netcdf4(year,var_list,unit_dict,frequency_list,fname,
                                       return_open_file=False,int32_list=[]):
    '''xarray sucks for netcdf
    '''
    # from netCDF4 import date2num,num2date -- use these with date2num(dt,'days since %i-01-01'%year) to convert!
    longitude = np.arange(4320) * 1/12 - 180 + 1/24
    latitude = -1.0*(np.arange(2160) * 1/12 - 90 + 1/24)

    ncfile = netCDF4.Dataset(fname,mode='w')#,format='NETCDF4')
    lat_dim = ncfile.createDimension('latitude', len(latitude))     # latitude axis
    lon_dim = ncfile.createDimension('longitude', len(longitude))    # longitude axis
    time1_dim = ncfile.createDimension('time1', None) # unlimited axis (can be appended to)
    time8_dim = ncfile.createDimension('time8', None) # unlimited axis (can be appended to)

    lat = ncfile.createVariable('latitude', np.float32, ('latitude',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = ncfile.createVariable('longitude', np.float32, ('longitude',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time1 = ncfile.createVariable('time1', np.float64, ('time1',))
    time1.units = 'days since %i-01-01'%year
    time1.calendar = 'proleptic_gregorian'
    time1.long_name = 'time1'
    time8 = ncfile.createVariable('time8', np.float64, ('time8',))
    time8.units = 'days since %i-01-01 00:00:00'%year
    time8.calendar = 'proleptic_gregorian'
    time8.long_name = 'time8'

    varlist = []
    for i,var in enumerate(var_list):
        if var in int32_list:
            v_dtype = np.int32
        else:
            v_dtype = np.float32

        if frequency_list[i] == 0:
            ncvar = ncfile.createVariable(var, v_dtype, ('latitude','longitude'),
                                          zlib=True,chunksizes=(360,len(longitude)))
        elif frequency_list[i] == 8:
            ncvar = ncfile.createVariable(var, v_dtype, ('time8','latitude','longitude'),
                                          zlib=True,chunksizes=(1,360,len(longitude)))
        elif frequency_list[i] == 1:
            ncvar = ncfile.createVariable(var, v_dtype, ('time1','latitude','longitude'),
                                          zlib=True,chunksizes=(1,360,len(longitude)))
        ncvar.write_freq = frequency_list[i]
        ncvar.units = unit_dict[var]
        varlist.append(ncvar)

    if return_open_file:
        return ncfile
    else:
        ncfile.close()


def build_run_params(param_dict=None, **kwargs):
    """ Contains a default set, with params to copy over/update
    inside a dict, or as key=values pairs

    param_dict : dictionary of new parameters:values
    **kwargs   : optional parameter_name=value pairs as method arguments
    """

    params = {
        # run logistics
        'run_name': 'test0',
        'output_path': r'C:\Users\blsaenz\Desktop',
        'fortran_calc': False,
        'reduced_output': False,   # only write annual data
        'monte_carlo_output': False,
        'start_year':  2003,
        'start_month':    1,
        'start_day':      1,
        'repeat_annual_forcing': False,
        'calc_steps':   365, # dimensionless
        'spinup_steps': 0,
        'mp_dt_mag':    1.0, # time step in days - don't expect this will change, may break things
        'code_path': r'X:\data\mag\magpy',
        'matlab_grid_filepath': r'X:\data\CWM\grid\cwm_grid.mat',
        'default_cwm_mask': r'X:/data/CWM/scripts/python/mask_cbpm_2021_01_13.txt',
        'default_cwm_grid_area': r'X:\data\CWM\grid\area_twelfth_degree.txt',
        'datetime_in_output': True,
        'seeding_type': 0,# 0=init only,1=monthly from forcing file
        'suppress_calc_print': False, # does not work currently

        # output & tracer variable frequency
        # -1=no output; 1=daily; 8=every 8 days; 0=annual;
        'B_freq':         8,
        'Q_freq':         8,
        'Gave_freq':      8,
        'Dave_freq':      8,
        'd_B_freq':       8,
        'd_Q_freq':       8,
        'Growth2_freq':   8,
        'd_Be_freq':      8,
        'd_Bm_freq':      8,
        'd_Ns_freq':      8,
        'harv_freq':      0,
        'GRate_freq':     1,
        'B_N_freq':       8,
        'n_harv_freq':    8,
        'min_lim_freq':    -1,
        'gQ_freq':    -1,
        'gT_freq':    -1,
        'gE_freq':    -1,
        'gH_freq':    -1,

        # species parameters [default is some version of Saccharina]
        'mp_spp_Vmax': 13.8*24,         # [umol N/g-DW/d], expecting umol N/m2/d, conversion below
        'mp_spp_Ks_NO3': 1400,        # [umol N/m3]  % L. Roberson
        'mp_spp_kcap': 3000.,         # [g(dry)/m]
        'mp_spp_Gmax_cap': 0.2,       # [1/day]
        'mp_spp_PARs': 70./4.57,       # [W/m2]
        'mp_spp_PARc': 23.4/4.57,      # [W/m2]
        'mp_spp_Q0': 32.0,            # initial Q [mg N/g(dry]
        'mp_spp_Qmin': 10.18,           # [mg N/g(dry)]
        'mp_spp_Qmax': 54.0,          # [mg N/g(dry)]
        'mp_spp_BtoSA': 1.0,          # Hmm not used right now???
        'mp_spp_line_sep': 0.7,       # lines per m i.e. 1.5 separation = 0.75
        'mp_spp_kcap_rate': 0.01 ,    # [1/day]
        'mp_spp_Topt1': 10.0,         # [deg C]
        'mp_spp_K1': 0.03,            # temp func slope 1
        'mp_spp_Topt2': 15.0,         # [deg C]
        'mp_spp_K2': 0.1,             # temp func slope 2
        'mp_spp_CD':0.10,             # drag coefficient (unitless)
        'mp_spp_dry_sa': 58.,         # [g(dry)/m2]
        'mp_spp_dry_wet': 0.094,      # [g(dry)/g(wet)] % Not changed from the macrocystis values
        'mp_spp_E': 0.005,            # [d-1] % No info specific for Eucheuma
        'mp_spp_seed': 50.0,          # initial biomass [g(dry)/m]
        'mp_spp_death': 0.01,         # death rate [1/day]

        # harvest parameters
        'mp_harvest_type': 0,         # 0 = harvest to seed weight, 1 = harvest mp_harvest_f fraction
        'mp_harvest_schedule': 0,     # 0 = fixed harvest freq from start/seed, 1 = conditional harvest
        'mp_harvest_avg_period': 7,   # days to average growth/death over, used by conditional harvesting
        'mp_harvest_kg': 1.0,         # kg to take if available , not used by all harvest types
        'mp_harvest_freq': 60,         # harvest frequency [days] , not used by all harvest types
        'mp_harvest_f': 0.6,          # harvest fraction
        'mp_harvest_span': 15,        # if conditional harvest, the +/1 span in days around ideal harvest date whhen harvest is possible
        'mp_harvest_nmax': 4,         # max harvests per year/harvest period

        # more parameters
        'mp_N_flux_limit': 1,         # 1(grater than 0) = limit growth using CESM N vertical flux; -1(or less than 0) = don't limit growth
        'mp_breakage_type': 0,
        'mp_dte': 1.0,                # matlab datenum of timestep [days since 2000-01-01]
        'mp_farm_depth':   1.0,       # meters
        'mp_wave_mort_factor': 1.0    # scaling factor for the wave mortality relationship
    }
    if param_dict is not None:
        for k,v in param_dict.items():
            params[k] = v
    for k,v in kwargs.items():
        params[k] = v

    return params

class MAG0_base(object):
    default_forcing_vars = ['sst','par','chl','swh','mwp','cmag','no3','nflux','seed'] # order is important/used w forcing vars
    tracer_vars = ['B', 'Q', 'Gave', 'Dave']  # variables that are carried as tracers (in/out during model calculation)

    output_vars = ['d_B', 'd_Q', 'Growth2', 'd_Be', 'd_Bm', 'd_Ns', 'harv', 'GRate', 'B_N', 'n_harv',
                   'min_lim', 'gQ', 'gT', 'gE', 'gH']  # full output variables

    mc_output_vars = ['Growth2', 'd_Be', 'd_Bm', 'd_Ns', 'harv', 'n_harv']
    cummulative_vars = ['harv', 'd_Bm', 'Growth2', 'd_Be', 'd_Ns', 'n_harv',
                        'B_N']  # these output vars are zeroed after writing [B_N is not cummulative, but sometimes we assigned other things to it in mag_calc....]
    instantaneous_vars = ['GRate', 'min_lin']  # these output vars are zeroed every time step
    reduced_write_vars = ['B', 'Growth2', 'harv']  # if the reduced write parameter is set, only these vars are written
    write_vars = tracer_vars + output_vars
    units = {'B': 'g DW / m2',  # current seaweed biomass
             'Q': 'mg N / g DW',  # current Q [nutrient status of seaweed]
             'd_B': 'g DW / m2',  # time step (instantaneous) change in B
             'd_Q': 'mg N / g DW',  # time step (instantaneous) change in Q
             'Growth2': 'g DW / m2',  # total growth of seaweed in cell
             'd_Be': 'mg N / m2',
             # exudation: will need to be stoichiometrically converted to something useful, like g C, in postprocessing
             'd_Bm': 'g DW / m2',  # biomass that died (not harvested)
             'd_Ns': 'mg N / m2',  # total nutrient uptake due to growth
             'harv': 'g DW / m2',  # harvest amount (cumulative)
             'GRate': '1 / day',  # daily growth rate
             'B_N': 'g DW / g N',  # ratio of biomass DW / biomass N
             'n_harv': '# harvests',  # number of harvests that occured in each pixel
             'Gave': '1 / day',  # exponential running mean of growth rate
             'Dave': '1 / day',  # exponential running mean of death/loss rate
             'min_lim': 'lim_var',  # instantaneous 1=gQ,2=gT,3=gE,4=gH, 0 = masked/no calc
             'gQ': 'fractional',  # nutrient growth limitation term
             'gT': 'fractional',  # temperature growth limitation term
             'gE': 'fractional',  # light growth limitation term
             'gH': 'fractional',  # crowding growth limitation term
             }
    integer_vars = ['n_harv', 'min_lim']

    grid = {}
    tracers = {}
    outputs = {}
    write_event = {}
    write_count = {}

    init_seeding_type = 0  # seed in model initiation only
    monthly_seeding_type = 1  # seed according to a month in seed forcing file

    month_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    def __init__(self, params, forcing_var_meta={}):
        # save options and get read to run loop
        self.p = params
        self.forcing_var_meta = forcing_var_meta

    @contextmanager
    def suppress_stdout(self):
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    def update_params(self, new_params):
        '''Update parameters, and now model can be re-run without instantiating a new MAG0 class.  Might be
        useful for incremental or sensitivity studies. Likely it would be useful to change run_name! '''
        for p, v in new_params.items():
            self.p[p] = v

    def build_fortran_params(self):
        # put calc parameters into array
        fenum = fortran_param_enum_()
        npar = len(fenum)
        # these two parameters are potentially dynamic - record enum here for quick reference later
        self.mp_dte = fortran_param_enum_('mp_dte')
        if self.p['fortran_calc']:
            fortran_params = np.zeros(npar, dtype=np.float32)
            self.mp_dte -= 1
            for rp, i in fenum.items():
                fortran_params[i - 1] = self.p[rp]
        else:
            fortran_params = np.zeros(npar + 1,
                                      dtype=np.float32)  # pad with one extra [0] element so fortran enum works in python
            for rp, i in fenum.items():
                print(rp)
                fortran_params[i] = self.p[rp]
        return npar,fortran_params

    def init_timing(self):
        print('Creating timing...')
        self.time_unit = 'days since %i-01-01' % self.p['start_year']
        self.start_dt = datetime.datetime(self.p['start_year'], self.p['start_month'], self.p['start_day'])
        self.dt_dt = datetime.timedelta(days=self.p['mp_dt_mag'])
        self.dt_steps = [self.start_dt + self.dt_dt * i for i in range(self.p['calc_steps'])]
        self.dn_steps = date2num(self.dt_steps, self.time_unit)

        # store some facts about our timing, useful for harvest/seeding calcs
        self.dt_year = np.array([d.year for d in self.dt_steps])
        self.dt_month = np.array([d.month for d in self.dt_steps])
        self.dt_day = np.array([d.day for d in self.dt_steps])
        self.dt_doy = np.array([d.timetuple().tm_yday for d in self.dt_steps])

    def compute(self):
        self.init_timing()

        # fortran needs babying
        npar,fortran_params = self.build_fortran_params()

        # allocate internal memory for mag tracers and computational variables
        self.init_memory()
        self.init_forcing()

        # deal with output options, storage, and saving run parameters
        self.build_output_vars()
        self.setup_and_create_output_directory()
        self.init_write()

        # setup data structures needed to evaluate seeding and harvesting schemes
        self.init_seeding()
        self.init_harvest()

        if self.p['suppress_calc_print']:
            self.suppress_stdout()

        # main model loop
        for i, dn in enumerate(self.dn_steps):

            print('Calc step ', i, '...')
            steptime = time.time()

            # load any new forcing
            stime = time.time()
            self.load_forcing(i)
            print('  forcing load time ...', time.time() - stime)

            # update_calc_date
            fortran_params[self.mp_dte] = date2num(self.dt_steps[i], 'days since 2000-01-01')

            # WTF - this is all confusing and needs simplification
            # if self.p['seeding_type'] == self.init_seeding_type:
            #     self.seed[...] = self.do_harvest[...]  # don't re-assign seed array, will get wiped in harvest calc
            #     self.check_harvest(i)
            # else:
            #     self.check_harvest(i)
            #     self.seed = self.find_seeding(i)

            self.check_harvest(i)
            if self.p['seeding_type'] != self.init_seeding_type:
                self.seed = self.find_seeding(i)

            self.external_pre_step_calc(i)

            # mag_calc needs timing in date since 2000 -- does it?
            stime = time.time()
            if self.p['fortran_calc']:

                # mag_calc(lat,lon,sst,par,swh,cmag,no3,nflux,mask, &
                # params, &
                # Q,B,d_B,d_Q,Growth2,d_Nf,d_Bn,d_Ns,harv,GRate,B_N)
                mag_calc_f(self.ylat, self.xlon360,
                           self.f0[0], self.f0[1], self.f0[2], self.f0[3], self.f0[4], self.f0[5], self.f0[6],
                           self.f0[7],
                           self.do_harvest, self.seed, self.mask, self.GD_count, self.outputs['n_harv'],
                           self.total_harvests,
                           fortran_params,  # parameter float array
                           self.tracers['Q'], self.tracers['B'], self.outputs['d_B'], self.outputs['d_Q'],
                           # in/out arrays
                           self.outputs['Growth2'], self.outputs['d_Be'], self.outputs['d_Bm'],
                           self.outputs['d_Ns'], self.outputs['harv'], self.outputs['GRate'],
                           self.outputs['B_N'], self.tracers['Gave'], self.tracers['Dave'], self.outputs['min_lim'])
            else:

                mag_calc(self.ylat, self.xlon360,
                         self.f0[0], self.f0[1], self.f0[2], self.f0[3], self.f0[4], self.f0[5], self.f0[6], self.f0[7],
                         self.do_harvest, self.seed, self.mask, self.GD_count, self.outputs['n_harv'],
                         self.total_harvests,
                         self.nx, self.ny, npar,  # integer inputs
                         fortran_params,  # parameter float array
                         self.tracers['Q'], self.tracers['B'], self.outputs['d_B'], self.outputs['d_Q'],
                         # in/out arrays
                         self.outputs['Growth2'], self.outputs['d_Be'], self.outputs['d_Bm'],
                         self.outputs['d_Ns'], self.outputs['harv'], self.outputs['GRate'],
                         self.outputs['B_N'], self.tracers['Gave'], self.tracers['Dave'], self.outputs['min_lim'],
                         # )#,
                         self.outputs['gQ'], self.outputs['gT'], self.outputs['gE'], self.outputs['gH'], )

            print('  calc time ...', time.time() - stime)

            self.external_post_step_calc(i)

            self.write_output(i)


        # return any data returned by write_close
        return self.close_write()

    def update_calc_date(self, i, fortran_params):
        fortran_params[self.mp_dte] = date2num(self.dt_steps[i], 'days since 2000-01-01')

    def init_memory(self):
        order = 'F' if self.p['fortran_calc'] else 'C'

        # tracer arrays
        for fv in self.tracer_vars:
            print('  Allocating', fv)
            if fv in self.integer_vars:
                self.tracers[fv] = np.full((self.nx, self.ny), 0, np.int32, order=order)
            else:
                self.tracers[fv] = np.full((self.nx, self.ny), np.nan, np.float32, order=order)

        # output arrays
        for fv in self.output_vars:
            print('  Allocating', fv, self.p[fv + '_freq'])
            if fv in self.integer_vars:
                self.outputs[fv] = np.full((self.nx, self.ny), 0, np.int32, order=order)
            else:
                self.outputs[fv] = np.full((self.nx, self.ny), np.nan, np.float32, order=order)

            # these need to be initialized to zero
            if fv in self.cummulative_vars:
                self.outputs[fv][...] = 0.0

        # internal arrays - maybe less confusing if these are created in init_seeding and init_harvest?()
        self.total_harvests = np.full((self.nx, self.ny), 0, np.int32,
                                      order=order)  # internal array to test against mp_harvest_nmax
        self.partial_seed = np.full((self.nx, self.ny), 0, np.int32, order=order)  # used if a seeding day is occuring
        self.do_not_seed = np.full((self.nx, self.ny), 0, np.int32, order=order)  # used on days when no seeding
        self.do_harvest = np.full((self.nx, self.ny), 0, np.int32,
                                  order=order)  # used to if there is dynamic/calculated harvesting
        self.seeding_doy = np.full((self.nx, self.ny), 0, np.int32,
                                   order=order)  # used to diff with current doy to calculate dynamic harvest
        self.GD_count = np.full((self.nx, self.ny), 0, np.int32, order=order)
        self.seed = np.full((self.nx, self.ny), 0, np.int32, order=order)
        self.mask = np.full((self.nx, self.ny), 0, np.int32, order=order)
        self.ylat = np.full((self.nx, self.ny), 0, np.int32, order=order)
        self.xlon360 = np.full((self.nx, self.ny), 0, np.int32, order=order)

    def build_output_vars(self):
        if self.p['reduced_output']:
            self.write_vars = self.reduced_write_vars

        # remove write vars that are turned off by setting var_freq == -1
        self.write_vars = [fv for fv in self.write_vars if self.p[fv + '_freq'] != -1]

        # montecarlo write options overide others
        if self.p['monte_carlo_output']:
            self.write_vars = ['Growth2', 'd_Be', 'd_Bm', 'd_Ns', 'harv', 'n_harv']
            for v in self.write_vars:
                self.p[v + '_freq'] = 0

    def setup_and_create_output_directory(self):

        print('Creating & opening output files...')
        # create output directory and files
        now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.run_label = self.p['run_name']
        if self.p['datetime_in_output']:
            self.run_label = self.run_label + '_' + now_str
        self.outdir = os.path.join(self.p['output_path'],self.run_label)
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        self.writefile_stub = os.path.join(self.outdir,'mag0_output_'+self.run_label)

        # copy code
        code_dir = os.path.join(self.outdir,'code')
        os.mkdir(code_dir)
        code_files = []
        for localname in os.listdir(self.p['code_path']):
            if localname.endswith('.f90') or localname.endswith('.py'):
                code_files.append(localname)
        for f in code_files:
            shutil.copy(os.path.join(self.p['code_path'],f),os.path.join(code_dir,f))

        # write option to txt and pickle
        self.p['run_label'] = self.run_label
        self.p['out_path'] = self.outdir
        self.p['params_csv'] = 'mag0_params_'+self.run_label+'.csv'
        self.p['forcing_txt'] = 'mag0_forcing_'+self.run_label+'.txt'
        self.p['params_pkl'] = 'mag0_params_'+self.run_label+'.pkl'
        w = csv.writer(open(os.path.join(self.outdir,self.p['params_csv']),"w",newline='', encoding='utf-8'))
        for key, val in self.p.items():
            w.writerow([key, val])

        with open(os.path.join(self.outdir,self.p['forcing_txt']), "w") as f:
            for key, nested in sorted(self.forcing_var_meta.items()):
                print(key, file=f)
                for subkey, value in sorted(nested.items()):
                    print('   {}: {}'.format(subkey, value), file=f)
                print(file=f)

        with open(os.path.join(self.outdir,self.p['params_pkl']),"wb") as fp:
            pickle.dump(self.p,fp)

    def check_harvest(self, nstep):

        # find how may days away from seeding+harvest_freq we currently are
        # why does harvest get determined by seeding? These names could be changed for clarity
        doy_diff = self.dt_doy[nstep] - self.seeding_doy
        doy_diff[doy_diff < 0] = doy_diff[
                                     doy_diff < 0] + 365  # convert so that doy_diff is positive relative to seeding
        doy_diff2 = np.copy(doy_diff)
        doy_diff[doy_diff == 0] = 1  # don't harvest on seed doy...
        doy_mod = np.mod(doy_diff, self.p['mp_harvest_freq'])

        self.do_harvest[...] = 0

        if self.p['mp_harvest_schedule'] == 0:
            # this could be done ahead of time ... but maybe not for multi-year runs?
            self.do_harvest[doy_mod == 0] = 1

        elif self.p['mp_harvest_schedule'] == 1:

            if self.p['mp_harvest_span'] < 0:
                # no harvest span -- harvest anytime kg trigger is reached
                self.do_harvest[...] = 1
            else:
                # if harvest schedule is conditional/flexible, find if we are in harvest span
                harvest_mask = doy_mod >= self.p['mp_harvest_freq'] - self.p['mp_harvest_span']
                self.do_harvest[harvest_mask] = 1  # within harvest span
                self.do_harvest[doy_mod == 0] = 2  # done - final harvest, cell will be masked from further calcs

        elif self.p['mp_harvest_schedule'] == 2:  # fixed within period
            harv_period = np.int32(np.float32(self.p['mp_harvest_span']) / self.p['mp_harvest_nmax'])
            self.do_harvest[doy_mod == 0] = 2  # done - final harvest, cell will be masked from further calcs
            for i in range(1, self.p['mp_harvest_nmax']):
                self.do_harvest[doy_diff2 == (self.p['mp_harvest_freq'] - harv_period * i)] = 1


    def init_forcing(self):
        raise NotImplementedError('init_forcing() not specified/implemented!')

    def init_write(self):
        '''Build and allocate self.output dictionary, containing output vars as keys to (nx,ny) size output arrays.'''
        raise NotImplementedError('init_output() not specified/implemented!')

    def load_forcing(self,nstep):
        raise NotImplementedError('load_forcing() not specified/implemented!')

    def write_output(self,nstep):
        raise NotImplementedError('write_output() not specified/implemented!')

    def init_harvest(self):
        raise NotImplementedError('init_harvest() not specified/implemented!')

    def find_seeding(self):
        raise NotImplementedError('find_seeding() not specified/implemented!')

    def init_seeding(self):
        raise NotImplementedError('init_seeding() not specified/implemented!')

    def external_pre_step_calc(self,nstep):
        '''Optional calculation that child model classes may implement'''
        pass

    def external_post_step_calc(self,nstep):
        '''Optional calculation that child model classes may implement'''
        pass

    # Not sure this should be in superclass
    def clear_cummulative_vars(self):
        # clear cummulative vars
        for fv in self.cummulative_vars:
            self.outputs[fv][...] = 0.0

    # Not sure this should be in superclass
    def clear_instantaneous_vars(self):
        # clear instantaneous vars
        for fv in self.instantaneous_vars:
            self.outputs[fv][...] = 0.0

    def close_write(self):
        raise NotImplementedError('close_output() not specified/implemented!')


class MAG0(MAG0_base):

    # grid is set to 1x1 for 0-D MACMODS
    nx = 1
    ny = 1

    def __init__(self, params, forcing_data,forcing_resample_type='interpolate'):
        '''forcing data is a pandas dataframe with datetimeindex and all of the forcing vars named. It should span
        the range of planned simulations, and will be interpolated to the timestep as needed.  Maybe think about
        whether linear interpolation is appropriate for all forcing vars?'''
        self.forcing_data = forcing_data
        self.forcing_resample_type = forcing_resample_type

        # do parent init()
        super(MAG0, self).__init__( params, {})


    def init_forcing(self):
        self.f0 = []
        for i, fv in enumerate(self.default_forcing_vars):
             self.f0.append(np.full((self.nx, self.ny), np.nan, np.float32, order="C"))

        # create self.finterp Dataframe from self.forcing_data in simulation time increments
        if self.p['mp_dt_mag'] == 1./24.:
            dt_str = 'H'
        elif self.p['mp_dt_mag'] == 1:
            dt_str = 'D'
        else:
            dt_str = '%iT'%int(1440*self.p['mp_dt_mag'])  # minutes

        # this could go wrong, if sub-sampling
        if self.forcing_resample_type == 'interpolate':
            finterp = self.forcing_data.resample(dt_str).interpolate()
        elif self.forcing_resample_type == 'mean':
            finterp = self.forcing_data.resample(dt_str).mean()
        elif self.forcing_resample_type == 'nearest':
            finterp = self.forcing_data.resample(dt_str).nearest()
        else:
            raise ValueError('forcing_resample_type not understood:',self.forcing_resample_type)
        # copy result sliceand garbage collect intermediate dataframe
        self.finterp = finterp[self.dt_steps[0]:self.dt_steps[-1]].copy()


    def init_write(self):
        self.write_store = {}
        for v in self.write_vars:
            self.write_store[v] = np.full(self.p['calc_steps'],np.nan,np.float32)

    def load_forcing(self,nstep):
        for i,v in enumerate(self.default_forcing_vars):
            self.f0[i][...] = self.finterp[v][nstep]
        self.ylat[...] = self.finterp['ylat'][nstep]
        self.xlon360[...] = self.finterp['xlon360'][nstep]

    def init_seeding(self):

        self.load_forcing(0)  # make sure forcing in loaded so we have initial NO3 value

        if self.p['seeding_type'] == self.init_seeding_type:
            # do the seeding here - don't do it dynamically within mag_calc
            self.tracers['B'][...] = self.p['mp_spp_seed']
            # set initial Q - is this maybe not needed anymore? Does it happen in mag_calc anyway?
            no3_f_var = self.default_forcing_vars.index('no3')
            self.tracers['Q'][...] = self.p['mp_spp_Qmin'] + self.f0[no3_f_var]*(self.p['mp_spp_Qmax']-self.p['mp_spp_Qmin'])/35.0
            self.mask[...] = 1
            #self.seeding_doy[...] = self.dt_doy[0] ! already seeded, don't re-seed in mag_calc?

        elif self.p['seeding_type'] == self.monthly_seeding_type:
            seed_f_var = self.default_forcing_vars.index('seed')
            self.seeding_month = self.f0[seed_f_var][0,0] # use first value - there can be only one seed month for monthly seeding
            self.seeding_doy[...] = self.month_doy[self.seeding_month - 1]

            # this is inane ... this functionality in place to save compute time in gMACMODS, seems not worth it
            self.do_seed = np.copy(self.seed)
            self.do_seed[...] = 1
            self.do_not_seed = np.copy(self.seed)
            self.do_not_seed[...] = 0

            self.mask[...] = 0

    def write_output(self,nstep):
        for v in self.write_vars:
            source = self.tracers if v in self.tracer_vars else self.outputs
            self.write_store[v][nstep] = source[v][0,0]
        self.clear_cummulative_vars()

    def find_seeding(self,nstep):
        """forcing specified a seeding month, check if month change and set seeding array & mask."""
        if self.p['seeding_type'] > 0:
            if self.p['seeding_type'] == self.monthly_seeding_type:    # monthly
                if self.dt_day[nstep] == 1: # turn of the month
                    if self.seed_month == self.dt_month[nstep]:
                        return self.do_seed
                    print('Seeding month:',self.forcing.month[nstep])
                    seed_mask = self.seed_mask(self.forcing.month[nstep])
                    self.partial_seed[...] = 0
                    self.partial_seed[seed_mask] = 1 # tell mag_calc to seed
                    self.mask[seed_mask] = 1 # update computation mask
                    return self.partial_seed
            # implement other seeding type, along with specified harvests??

        return self.do_not_seed

    def init_harvest(self):
        # reset cumulative var
        self.outputs['harv'][...] = 0.0


    def close_write(self):
        # write out harvest to dataframe, nc?
        self.write_store['date'] = pd.to_datetime(self.dt_steps)
        df = pd.DataFrame(self.write_store).set_index('date')
        df = df.merge(self.finterp,left_index=True,right_index=True)
        # write csv output
        df.to_csv(self.writefile_stub+'.csv')

        # write netcdf
        ds = xr.Dataset.from_dataframe(df)
        # add variable attribute metadata
        encoding = {}
        for v in self.output_vars:
            ds[v].attrs['units'] = self.units[v]
            ds[v].attrs['long_name'] = 'fake_long_name'
            dtpe = np.int32 if v in self.integer_vars else np.float32
            encoding[v] = {"dtype":dtpe,"zlib": True, "complevel": 4}

        # add global attribute metadata
        ds.attrs={'Conventions':'CF-1.6', 'title':'MAG0 model output', 'summary':''}
        # save to netCDF
        ds.to_netcdf(self.writefile_stub+'.nc',mode='w',format='netcdf4',encoding=encoding)

        return df

class gMACMODS(MAG0_base):

    def __init__(self, params, forcing_var_meta={}):

        # do parent init()
        super(gMACMODS, self).__init__( params, forcing_var_meta)

        # read grid
        self.load_grid()

    def init_forcing(self):
        order='F' if self.p['fortran_calc'] else 'C'
        if order == 'F':
            self.mask_F = np.full((self.nx,self.ny),0,np.int32,order=order)
            self.mask_F[...] = self.mask
            self.mask = self.mask_F

        # forcing arrays
        print('Allocating and opening forcing...')
        self.forcing = gMACMODS_forcing(self.dt_steps,self.forcing_var_meta,
                                    self.p['fortran_calc'],self.p['repeat_annual_forcing'])
        self.fplist = self.forcing.open_forcing(0)
        self.f0 = []
        self.seed_f_var = self.forcing.vars.index('seed')
        # self.f1 = []  # second forcing set for alternative loading?
        for i,fv in enumerate(self.forcing.vars):
            nx = self.fplist[i][4]
            ny = self.fplist[i][5]
            self.f0.append( np.full((nx,ny),np.nan,np.float32,order="C"))
            #self.f1.append( np.full((self.nx,self.ny),np.nan,np.float32,order=order) )


    def load_forcing(self,nstep):
        self.fplist = self.forcing.load_forcing(nstep, self.f0, fplist=self.fplist)

    def init_write(self):

        # save spinup steps as var for direct/non-dictionary reference
        self.write_start = self.p['spinup_steps']

        for fv in self.write_vars:
            self.write_event[fv] = step_event_from_freq(np.arange(self.p['calc_steps']),
                                                        self.p[fv+'_freq'],
                                                        max_steps=self.p['calc_steps'])
            self.write_count[fv] = 0
            # always write out tracer/cummulative vars after last step calc
            if fv in self.tracer_vars:
                self.write_event[fv][-1] = np.max(self.write_event[fv])+1  # adding next write to last tstep
            if fv in self.cummulative_vars:
                self.write_event[fv][-1] = np.max(self.write_event[fv])+1  # adding next write to last tstep

        self.create_open_output_nc(0)

    def load_grid(self):
        # read matlab struct
        print('Reading grid...')
        # we don't seem to need this...
        # self.grid={}
        # with h5py.File(self.p['matlab_grid_filepath'],'r') as f:
        #     for gname in f['cwm_grid'].keys():
        #         self.grid[gname] = f['cwm_grid'][gname][...]
        self.longitude,self.latitude,self.xlon,self.ylat,self.mask = load_cwm_grid(self.p['default_cwm_mask'])
        self.ny = len(self.longitude)
        self.nx = len(self.latitude)
        self.longitude360 = self.longitude
        self.longitude360[self.longitude360<0.0] = self.longitude360[self.longitude360<0.0] + 360.0
        self.xlon360,_ = np.meshgrid(self.longitude360, self.latitude)

        # mask seems backwards, and bool
        self.mask = np.logical_not(self.mask).astype(np.int32)
        self.ocean_mask = self.mask.astype(np.bool)


    def create_open_output_nc(self,step):
        year = self.dt_steps[step].year
        freq_list = [self.p[fv+'_freq'] for fv in self.write_vars]
        self.nc_outfile = self.writefile_stub + ".nc"
        if not os.path.exists(self.nc_outfile):
            # below did not work because xarray creates the whole file in memory before writing
            #out_dset = create_mag0_output_dataset(year,self.output_vars,self.output_units,freq_list)
            #encoding = {v:{"dtype":np.float32,"zlib": True, "complevel": 4, "chunksizes":(1,360,4320)} for v in self.output_vars}
            #out_dset.to_netcdf(self.nc_outfile,mode='w',format='netcdf4',encoding=encoding)
            create_gMACMODS_output_dataset_netcdf4(year,self.write_vars,self.units,freq_list,
                                               self.nc_outfile,int32_list=self.integer_vars)

        self.nc_rg = netCDF4.Dataset(self.nc_outfile,'a')
        self.nc_rg['latitude'][...] = self.latitude
        self.nc_rg['longitude'][...] = self.longitude
        self.nc_rg.sync()


    def write_output(self,nstep):

        if nstep >= self.write_start:

            if nstep == self.write_start:
                self.clear_cummulative_vars()

            if nstep == self.write_start-1:
                self.clear_cummulative_vars()
                self.clear_instantaneous_vars()

            if nstep >= self.write_start:

                #stime = time.time()

                dn = date2num(self.dt_steps[nstep],self.time_unit)

                for fv in self.write_vars:
                    if self.write_event[fv][nstep] >= 0:
                        print('writing output:',fv,'...')
                        write_freq = self.nc_rg.variables[fv].write_freq
                        source = self.tracers if fv in self.tracer_vars else self.outputs
                        #if self.p['fortran_calc']:
                        #    write_arr = ascontiguousarray(source[fv])
                        #else:
                        write_arr = source[fv]

                        if write_freq > 0:
                            self.nc_rg.variables['time%i'%write_freq][self.write_count[fv]] = dn  # repeated, could optimize
                            self.nc_rg.variables[fv][self.write_count[fv],:,:] = write_arr
                        else:
                            self.nc_rg.variables[fv][...] = write_arr  # no time; once per run

                        self.write_count[fv] += 1

                        # clear cummulative vars
                        if fv in self.cummulative_vars: # other cummulative vars?
                            source[fv][...] = 0.0



            #self.nc_rg.sync()
            #print('  write time ...',time.time()-stime)


    def init_Q(self):
        """Go through the motions of loading 1st forcing just to read no3, and
        set initial Q. Make sure to call self.init_memory() before this."""
        fforce = self.forcing.load_forcing(0,self.f0,fplist=self.fplist)
        no3_f_var = self.forcing.vars.index('no3')
        return self.p['mp_spp_Qmin'] + self.f0[no3_f_var]*(self.p['mp_spp_Qmax']-self.p['mp_spp_Qmin'])/35.0

    def init_seeding(self):
        if self.p['seeding_type'] == self.init_seeding_type:
            # do the seeding here - don't do it dynamically within mag_calc
            self.tracers['B'][self.ocean_mask] = self.p['mp_spp_seed']
            self.tracers['Q'][self.ocean_mask] = self.init_Q()[self.ocean_mask]
            self.mask[self.ocean_mask] = 1

        elif self.p['seeding_type'] == self.monthly_seeding_type:    # monthly
            # mask everything initially, to skip computation until 1st seeding
            self.mask[...] = 0

    def seed_mask(self,seed_indicator):
        seed_mask = self.f0[self.seed_f_var] == seed_indicator
        return np.logical_and(seed_mask,self.ocean_mask)

    def find_seeding(self,nstep):
        """If using seeding file, check if month change and set seeding array & mask."""
        if self.p['seeding_type'] > 0:
            if self.p['seeding_type'] == self.monthly_seeding_type:    # monthly
                if self.forcing.day[nstep] == 1: # turn of the month
                    print('Seeding month:',self.forcing.month[nstep])
                    seed_mask = self.seed_mask(self.forcing.month[nstep])
                    self.partial_seed[...] = 0
                    self.partial_seed[seed_mask] = 1 # tell mag_calc to seed
                    self.mask[seed_mask] = 1 # update computation mask
                    return self.partial_seed

        return self.do_not_seed

    def init_harvest(self):
        # set seeding day-of-year, for later refernce during harvest determination

        # why does harvest get determined by seeding? These names could be changed for clarity

        if self.p['seeding_type'] == self.monthly_seeding_type:
            # load forcing to grab seeding month
            self.forcing.load_forcing(0,self.f0,fplist=self.fplist)
            for month in range(1,13):
                seed_mask = self.seed_mask(month)
                self.seeding_doy[seed_mask] = self.month_doy[month-1]
        elif self.p['seeding_type'] == self.init_seeding_type:
            self.seeding_doy[...] = self.forcing.doy[0]
        # reset cumulative var
        self.outputs['harv'][...] = 0.0

    def close_write(self):
        self.nc_rg.close()


class gMACMODS_forcing(object):

    default_vars = ['sst','par','chl','swh','mwp','cmag','no3','nflux','seed']
    default_meta = { 'sst': { 'freq': 8,
                              'path': r'X:\data\CWM\SST',
                              'fname': r'sst_YyYy.nc',
                              'h5_path': 'sst',
                             },
                     'par': { 'freq': 8,
                              'path': r'X:\data\CWM\PAR',
                              'fname': r'par_YyYy.nc',
                              'h5_path': 'par',
                              },
                     'chl': { 'freq': 8,
                              'path': r'X:\data\CWM\chla',
                              'fname': r'chl_YyYy.nc',
                              'h5_path': 'chl',
                              },
                     'swh': { 'freq': 8,
                              'path': r'X:\data\CWM\ECMWF',
                              'fname': r'ecmwf_cwm_YyYy_9km.nc',
                              'h5_path': 'swh_mean',
                              },
                     'mwp': { 'freq': 8,
                              'path': r'X:\data\CWM\ECMWF',
                              'fname': r'ecmwf_cwm_YyYy_9km.nc',
                              'h5_path': 'mwp_mean',
                              },
                     'cmag': { 'freq': 8,
                              'path': r'X:\data\CWM\hycom',
                              'fname': r'HYCOM_YyYy_9km.nc',
                              'h5_path': 'speed_mean',
                              },
                     'no3': { 'freq': 8,
                              'path': r'X:\data\CWM\2003_data_copy',
                              'fname': r'cwm_CESM_8day_interp_NO3_20m.nc',
                              'h5_path': 'NO3_20m',
                              },
                     'nflux': { 'freq': 26,
                              'path': r'X:\Globus',
                              'fname': r'cwm_CESM_5day_NO3_100m_Wflux_6_pixel.nc',
                              'h5_path': 'NO3_100m_Wflux_26_day',
                              },
                     'seed': { 'freq': 0,
                              'path': r'X:\data\mag\output\seeding_flux_limited',
                              'fname': r'Eucheuma_seed_month_60pix.nc',
                              'h5_path': 'seed_month',
                              },
    }

    def __init__(self, dt_steps,var_meta={},fortran_calc=False,
                 repeat_annual_forcing=False):
        '''vars: list containing and (extra) forcing variables
        freq, path are dicts with vars as keys, for updating load frequency and
        paths to the var folders.
        '''

        self.vars = self.default_vars
        self.meta = self.default_meta
        for v,m in var_meta.items():
            if not v in self.vars:
                self.vars.append(v)
            self.meta[v] = m
        self.nvars = len(self.vars)

        self.fortran_calc = fortran_calc
        self.repeat_annual_forcing = repeat_annual_forcing

        self.year = np.array([d.year for d in dt_steps])
        self.month = np.array([d.month for d in dt_steps])
        self.day = np.array([d.day for d in dt_steps])
        self.doy = np.array([d.timetuple().tm_yday for d in dt_steps])

        # hack to prevent doy 366
        # while np.any(self.doy==366):
        #     idx = np.where(self.doy==366)[0][0]
        #     self.doy[idx:-2] = self.doy[idx+1:-1]
        #     self.doy[-1] += 1
        #     if self.doy[-1] == 366:
        #         self.doy[-1] = 365

        # if files are open, save them here
        self.fplist = None

        # par is specieal - we want to cleanse NANs, so save index
        self.par_idx = self.vars.index('par') # store for later cleansing of nans
        self.chl_idx = self.vars.index('chl') # store for later cleansing of nans


    def open_forcing(self,nstep):
        """The fplist could be used by another copy of gMACMODS_forcing!
        fplist elements are tuple of (path, year, h5_file_pointer, dset, nx, xy)"""
        print('Opening forcing files...')
        self.fplist = []
        self.f_load = [] # list in order to self.var for speed (so not dict lookups)
        self.current_year = self.year[nstep]
        for v in self.vars:
            ffile = os.path.join(self.meta[v]['path'],self.meta[v]['fname'].replace('YyYy', '%i'%self.current_year))
            print('\tforcing file:',ffile)
            # check to see if file is aready open
            fp = None
            for fpl1 in self.fplist:
                if ffile == fpl1[0]:
                    fp = fpl1[2]
                    break
            if fp is None:
                fp = h5py.File(ffile, 'r')
            dset = fp[self.meta[v]['h5_path']]
            ny = dset.shape[-1]  # time var seems to be first
            nx = dset.shape[-2]
            if len(dset.shape) > 2:
                nt = dset.shape[-3]
            else:
                nt = 0
            self.f_load.append(self.generate_f_load(v, 365, nt))

            self.fplist.append((ffile,self.current_year,fp,dset,nx,ny,nt))
        return self.fplist

    def load_forcing(self,nstep,pre_alloc_arrays,fplist=None): #,fforce=None):
        """How to do this the fastest?  I think we want no dictionary lookups if possible, so cycle
        over a list of open file pointers/netcdf4 objects, update slices, input slices? Should the
        list of netcdf4 (or h5py?) objects be external, so that we can have two copies of this object
        around for simultaneous reads/compute?"""

        fpl = self.fplist if fplist is None else fplist
        if fpl is None:
            fpl = self.open_forcing(nstep)
        else:
            if not self.repeat_annual_forcing:
                if fpl[0][1] != self.year[nstep]:
                    # new year, open new forcing files
                    fpl = self.open_forcing(nstep)
        for i in range(self.nvars):
            path,yr,fp,dset,nx,ny,nt = fpl[i]
            f_load = self.f_load[i][nstep]
            if f_load >= 0:
                if f_load >= nt:
                    print('  requested time index %i, reducing to max time index (%i)'%(f_load,nt-1))
                    f_load = nt-1
                print('  loading',path,f_load)
                #dset.read_direct(pre_alloc_arrays[i], source_sel=np.s_[f_load,:,:], dest_sel=np.s_[0,:,:])
                if nt == 0:
                    ncslice = np.s_[:,:]
                else:
                    ncslice = np.s_[f_load,:,:]
                dset.read_direct(pre_alloc_arrays[i], source_sel=ncslice)

                # purge/cleanse NANs from par, chl to some defaults to allow computation
                # at high latitudes
                if i == self.par_idx:
                    pre_alloc_arrays[i][np.isnan(pre_alloc_arrays[i])] = 0.01
                    pre_alloc_arrays[i][pre_alloc_arrays[i] == -999.9] = 0.01
                    pre_alloc_arrays[i][pre_alloc_arrays[i] < 0] = 0.01
                if i == self.chl_idx:
                    pre_alloc_arrays[i][np.isnan(pre_alloc_arrays[i])] = 0.02
                    pre_alloc_arrays[i][pre_alloc_arrays[i] == -999.9] = 0.02
                    pre_alloc_arrays[i][pre_alloc_arrays[i] < 0] = 0.02

            #if self.fortran_calc:
            #    fforce[i] = ascontiguousarray(pre_alloc_arrays[i])
        return fpl

    def generate_f_load(self,var,max_steps,max_count):
        """ This is a special step event, because it is date-related, and the >= 0
        value is also the time slice for loading.
        """
        f_load = -1.0*np.ones(len(self.doy),np.int)
        if self.meta[var]['freq'] == 0:
            f_load[0] = 0
        else:
            annual_load = step_event_from_freq(np.arange(0,366),self.meta[var]['freq'],max_steps=max_steps,max_count=max_count)
            load_doy = np.where(annual_load >= 0)[0]  # days of year to load
            last_load = -1
            for i,d in enumerate(self.doy-1): # doy -1 because annual_load starts from 0
                if d in load_doy and last_load != d:
                    f_load[i] = np.int(annual_load[d])
                    if last_load == -1 and i > 0: # figure out load for 1st step, if between loads
                        f_load[0] = max(0,np.int(annual_load[d]-1))
                    last_load = d
        return f_load.astype(np.int)

def fortran_param_enum_(p=None):
    p_enum = {
        'mp_spp_Vmax' : 1   ,
        'mp_spp_Ks_NO3' : 2   ,
        'mp_spp_kcap' : 3   ,
        'mp_spp_Gmax_cap' : 4   ,
        'mp_spp_PARs' : 5   ,
        'mp_spp_PARc' : 6   ,
        'mp_spp_Q0' : 7   ,
        'mp_spp_Qmin' : 8   ,
        'mp_spp_Qmax' : 9   ,
        'mp_spp_BtoSA' : 10   ,
        'mp_spp_line_sep' : 11   ,
        'mp_spp_kcap_rate' : 12   ,
        'mp_spp_Topt1' : 13   ,
        'mp_spp_K1' : 14   ,
        'mp_spp_Topt2' : 15   ,
        'mp_spp_K2' : 16   ,
        'mp_spp_CD' : 17   ,
        'mp_spp_dry_sa' : 18   ,
        'mp_spp_dry_wet' : 19   ,
        'mp_spp_E' : 20   ,
        'mp_spp_seed' : 21   ,
        'mp_spp_death' : 22   ,
        'mp_harvest_type' : 23   ,
        'mp_harvest_schedule': 24,
        'mp_harvest_avg_period': 25,
        'mp_harvest_kg' : 26     ,
        'mp_harvest_f' : 27     ,
        'mp_harvest_nmax' : 28     ,
        'mp_breakage_type' : 29     ,
        'mp_dte' : 30            ,
        'mp_dt_mag' : 31            ,
        'mp_N_flux_limit' : 32     ,
        'mp_farm_depth' : 33     ,
        'mp_wave_mort_factor' : 34     ,
    }
    if p is None:
        return p_enum
    else:
        return p_enum[p]

@jit(nopython=True)
def mag_calc(lat,lon,sst,par,chl,swh,mwp,cmag,NO3,nflux, # input arrays
             do_harvest,seed_now,mask,GD_count,n_harv,t_harv, # input int arrays
             nx,ny,npar,  # integer inputs
             params, # parameter float array
             Q,B,d_B,d_Q,Growth2,d_Be,d_Bm,d_Ns,harv,GRate,B_N,Gave,Dave, # in/out arrays
             #min_lim): #,gQout,gTout,gEout,gHout): # in/out arrays
             min_lim,gQout,gTout,gEout,gHout): # in/out arrays


    #np.seterr(divide='raise')

    # sticking enum in here for now
    mp_spp_Vmax = 1
    mp_spp_Ks_NO3 = 2
    mp_spp_kcap = 3
    mp_spp_Gmax_cap = 4
    mp_spp_PARs = 5
    mp_spp_PARc = 6
    mp_spp_Q0 = 7
    mp_spp_Qmin = 8
    mp_spp_Qmax = 9
    mp_spp_BtoSA = 10
    mp_spp_line_sep = 11
    mp_spp_kcap_rate = 12
    mp_spp_Topt1 = 13
    mp_spp_K1 = 14
    mp_spp_Topt2 = 15
    mp_spp_K2 = 16
    mp_spp_CD = 17
    mp_spp_dry_sa = 18
    mp_spp_dry_wet = 19
    mp_spp_E = 20
    mp_spp_seed = 21
    mp_spp_death = 22
    mp_harvest_type = 23
    mp_harvest_schedule = 24
    mp_harvest_avg_period = 25
    mp_harvest_kg = 26
    mp_harvest_f = 27
    mp_harvest_nmax = 28
    mp_breakage_type = 29
    mp_dte = 30
    mp_dt_mag = 31
    mp_N_flux_limit = 32
    mp_farm_depth = 33
    mp_wave_mort_factor = 34

    # some internal params
    c0 = 0.0
    c1 = 1.0
    mw_n = 14.00672
    breakage_Duarte_Ferreira = 0             # fancy breakage
    breakage_Rodrigues       = 1             # death rate scales with swh

    # whole model domain calculations for current timestep
    # ------------------------------------------------------------
    KsNO3 = params[mp_spp_Ks_NO3]
    PARs = params[mp_spp_PARs]
    PARc = params[mp_spp_PARc]
    Qmax = params[mp_spp_Qmax]
    Qmin = params[mp_spp_Qmin]

      # !$OMP PARALLEL &
      # !$OMP DEFAULT(SHARED) &
      # !$OMP PRIVATE(i,j,thread,day_h,lambda,vQ,vNuTw_NO3,Uptake_NO3,UptakeN,gQ,gT,gE,gH,Growth,WP,M_Wave,M,f_harvest)
      # !$OMP DO SCHEDULE(DYNAMIC,chunk_size)
      #
      # ! ----------------------------------------------------------------

      # debuging range - loops switched to match Python

    #for i in range(1707,1709):
    #for i in range(1000,1001):
    #for i in range(375,376):
    #for i in range(275,276):
    for i in range(nx):
        #print('   calc col', i)
        #for j in range(0,30):
        #for j in range(1000,1001):
        #for j in range(425,426):
        #for j in range(2130,2131):
        for j in range(ny):
            #print('   calc row', j)
            #if j==2471:
            #    dude = 1

            #print('B',B[i,j],'sst',sst[i,j],'par',par[i,j])
            #print('no3',NO3[i,j],'swh',swh[i,j],'cmag',cmag[i,j])
            #print('nflux',nflux[i,j],'Q',Q[i,j])

            if mask[i,j] > 0: # and bad_forcing==0:

                if seed_now[i,j] > 0:
                    B[i,j] = params[mp_spp_seed]
                    Q[i,j] = Qmin + NO3[i,j]*(Qmax-Qmin)/35.0
                    t_harv[i,j] = 0 # reset internal harvest counter

                # Nutrient Uptake
                # ----------------------------------------------------------
                lambda0 = lambda_NO3(cmag[i,j],mwp[i,j],params[mp_spp_CD],params[mp_spp_Vmax],
                                     params[mp_spp_Ks_NO3],NO3[i,j])
                #print(j+1,lambda0)
                # Quota-limited uptake: maximum uptake when Q is minimum and
                # approaches zero as Q increases towards maximum; Possible that Q
                # is greater than Qmax. Set any negative values to zero.
                vQ = (Qmax-Q[i,j])/(Qmax-Qmin)
                vQ = max(c0,vQ)
                vQ = min(c1,vQ)

                # new vQ formulation that permits high uptake even with high Q, so there is not negative feedback
                # between high growth rate/high Q usage and reduced growth rate
                # The reasoning behind this uptake curve is that seaweed would not reduce updake under high-growth,
                # high-nutrient conditions, just because stores are full - it's really a time stepping issue. High Q->
                # leads to nutrient limitation in one time-step.  If at moderate Q, seaweeds are also prevented from
                # uptaking short-pulsed nutrients, like closer to the scale of the timestep.  Basically the linear
                # function above does not allow response to changing nutrients at rates that seaweeds are
                # capable of.
                #vQ = 1.0 - 1.0/(1.0+(Qmax-Q[i,j])*50.0/(Qmax-Qmin))
                #vQ = max(c0,vQ)
                #vQ = min(c1,vQ)

                # Below is what we call "Uptake Factor." It varies between 0
                # and 1 and includes kinetically limited uptake and
                # mass-transfer-limited uptake (oscillatory + uni-directional flow)
                NO3_u = NO3[i,j]*1000.0
                vNuTw_NO3 = NO3_u / (KsNO3 * ((NO3_u/KsNO3) + 0.5 * (lambda0+math.sqrt(lambda0**2 + 4.0 * (NO3_u /KsNO3)))))
                vNuTw_NO3 = max(c0,vNuTw_NO3)
                vNuTw_NO3 = min(c1,vNuTw_NO3)

                # Uptake Rate [mg N/g(dry)/d]
                # Nutrient Uptake Rate = Max Uptake * v[Ci,u,Tw] * vQ
                # converted from umol N/m2/d -> mg N/g(dry)/d by 14.0067 / 1e3
                Uptake_NO3 = params[mp_spp_Vmax] * vNuTw_NO3 * vQ # [umol/m2/d]
                Uptake_NO3 = Uptake_NO3 * mw_n / 1.e3
                UptakeN = Uptake_NO3
                UptakeN = UptakeN / params[mp_spp_dry_sa]

                # Growth
                # ----------------------------------------------------------
                # Growth, nitrogen movement from Ns to Nf = umax*gQ*gT*gE*gH; [per day]
                # Output:
                #   Growth, [h-1]
                #   gQ, quota-limited growth
                #       from Wheeler and North 1980 Fig. 2
                #   gT, temperature-limited growth
                #       piecewise approach taken from Broch and Slagstad 2012 (for sugar
                #       kelp) and optimized for Macrocystis pyrifera
                #   gE, light-limited growth
                #       from Dean and Jacobsen 1984
                #   gH, carrying capacity-limited growth

                # nutrient (quota-based) limitation
                #gQ = (Q[i,j] - Qmin) / Q[i,j] # Droop equation

                gQ = (Q[i,j] - Qmin) / Q[i,j] * Qmax/(Qmax-Qmin) # Droop scaled from 0-1

                #gQ = (Q[i,j] - Qmin) / (Qmax-Qmin) #Freider et al.

                gQ = max(c0,gQ)
                gQ = min(c1,gQ)

                # temperature limitation
                gT = temp_lim(sst[i,j],params[mp_spp_Topt1],params[mp_spp_K1],
                     params[mp_spp_Topt2],params[mp_spp_K2])

                # light limitation
                par_watts = par[i,j] * 2.515376387217542
                # attentuation according to MARBL
                chlmin = max(0.02,chl[i,j])
                chlmin = min(30.0,chlmin)
                if chlmin < 0.13224:
                    atten = -0.000919*(chlmin**0.3536)  # 1/cm
                else:
                    atten = -0.001131*(chlmin**0.4562)  # 1/cm
                par_watts = par_watts * math.exp(atten*params[mp_farm_depth]*100.0)

                if par_watts < PARc:
                    gE = c0
                # gMACMODS ------------------------------------
                elif par_watts > PARs:
                    gE = c1
                else:
                    gE = (par_watts-PARc)/(PARs-PARc)*math.exp(-(par_watts-PARc)/(PARs-PARc)+c1)
                # Frieder et al. 2022 -------------------------
                #else:
                #    gE = c1 - math.exp(-0.333*(par_watts-PARc))

                # consider daylength if timestep is > 1/2 a day
                if params[mp_dt_mag] > 0.51:
                    day_h = daylength(lat[i,j],lon[i,j],c0,params[mp_dte]) # daylength in h
                    gE = min(day_h,gE*day_h)


                # Carrying capacity
                # ----------------------------------------------------------
                # gH -> density-limited growth (ranges from the max growth rate to the death rate, excluding wave mortality)
                # This expression follows Xiao et al (2019 and ignores wave mortality when
                # thinking about the death rate

                Bnew = B[i,j] #params[mp_spp_line_sep] # converting from g/m2 to g/m

                kcap_slope = -0.75
                #kcap_slope = -1.44

                A = params[mp_spp_kcap_rate]/(params[mp_spp_kcap]**(kcap_slope))
                gH = A*Bnew**(kcap_slope)

                # below is confusing programatically - gH should not include Gmax - but this is what was used in gMACMODS < 2022-10
                #if gH > params[mp_spp_Gmax_cap]:
                #    gH = params[mp_spp_Gmax_cap]
                #Growth = gH * gT * gE * gH

                # interacting limitation terms growth model
                Growth = min(gH,params[mp_spp_Gmax_cap]) * gT * gE * gQ

                # independent light and nutrient limitation growth model
                #Growth = min(gH,params[mp_spp_Gmax_cap]) * gT * min(gQ,gE)

                gQout[i,j] = gQ
                gTout[i,j] = gT
                gEout[i,j] = gE
                gHout[i,j] = gH

                #           1, 2, 3, 4
                mlim = min([gQ,gT,gE,gH])
                if gT == mlim:  # depending on floating point type, this may not work in fortran
                    min_lim[i,j] = 2
                elif gE == mlim:
                    min_lim[i,j] = 3
                elif gQ == mlim:
                    min_lim[i,j] = 1
                elif gH == mlim:
                    min_lim[i,j] = 4


                # Mortality
                # ----------------------------------------------------------
                # d_wave = frond loss due to waves; dependent on Hs, significant
                # wave height [m]; Rodrigues et al. 2013 demonstrates linear relationship
                # between Hs and frond loss rate in Macrocystis [d-1] (continuous)
                # Duarte and Ferreira (1993) find a linear relationship between wave power and mortality in red seaweed.
                if params[mp_breakage_type] == breakage_Duarte_Ferreira:
                    #WP = rho.*g.^2/(64*pi)*swh.^2.*Tw
                    WP = 1025.0*9.8**2 / (64.0*math.pi)*swh[i,j]**2 * mwp[i,j] /1.e3 # [kW]
                    # [Duarte and Ferreira (1993), in daily percentage]
                    M_wave = (2.3*1e-4*WP + 2.2*1e-3) * params[mp_wave_mort_factor]
                elif params[mp_breakage_type] == breakage_Rodrigues:
                    # Rodrigues et al. 2013
                    M_wave = (params[mp_spp_death] * swh[i,j]) * params[mp_wave_mort_factor]

                # M = M_wave + general Mortality rate; [d-1]
                M = params[mp_spp_death] + M_wave

                # limit growth to nflux
                # ----------------------------------------------------------
                # Comparing the nitrogen taken up to the amount of nitrogen fluxed in per m2
                dNs = UptakeN*params[mp_dt_mag]*B[i,j]   # [mg-N/ m2]
                if params[mp_N_flux_limit] > c0:
                    N_new = nflux[i,j] * mw_n * 864.0              # /100 * 86400 * 14.006 [microM/cm2/s]->[mmol-N/m2/day] -> [mg-N/m2/day]
                    if dNs > N_new and N_new > c0:
                        dNs = N_new
                    elif N_new < c0:
                        dNs = c0

                # Redefining Uptake
                UptakeN = dNs/params[mp_dt_mag]/B[i,j] # [mg-N/g-dry/day]
                # Uptake2 = dNs # [mg-N] --> saving for post-processing

                # find exudation
                dE = params[mp_spp_E] * (Q[i,j]-Qmin) * params[mp_dt_mag] # mg N / gDW / m2

                # Output terms
                # ----------------------------------------------------------
                d_B[i,j] = Growth * B[i,j] * params[mp_dt_mag] - M * B[i,j] * params[mp_dt_mag]

                # gMACMODS d_Q
                #d_Q[i,j] = UptakeN * params[mp_dt_mag] - Growth * (Q[i,j]-Qmin) * params[mp_dt_mag] - dE

                # dilution total N d_Q
                d_Q_growth = Q[i,j] * ( 1.0/(1.0 + Growth*params[mp_dt_mag]) - 1.0)
                d_Q[i,j] = UptakeN * params[mp_dt_mag] + d_Q_growth - dE

                #print(j+1,Q[i,j],d_Q[i,j],B[i,j],d_B[i,j],Growth,gQ,gT,gE,gH,UptakeN)
                #print(j+1,par[i,j],day_h,gE,d_Q[i,j],UptakeN,lambda0,NO3_u,vNuTw_NO3)

                GRate[i,j] = Growth
                Growth2[i,j] += Growth * B[i,j] * params[mp_dt_mag]

                d_Be[i,j] += dE * B[i,j]
                d_Bm[i,j] += B[i,j] * M * params[mp_dt_mag]
                d_Ns[i,j] += UptakeN * params[mp_dt_mag] * B[i,j] # mg N per m2


                # write out non-standard variables, ovveriding what is supposed to be in output
                # ----------------------------------------------------------

                # uncomment to report breakage death , in place of B_N
                #B_N[i,j] += B[i,j] * M_wave * params[mp_dt_mag]
                B_N[i,j] = 1.e3/Q[i,j] #--- this is what B_N is supposed to be, uncommented for standard output

                # uncomment to report biomass used by limitation functions, in place of cummulative growth
                #Growth2[i,j] = B[i,j]



                # Update State Variables
                # ----------------------------------------------------------

                B[i,j] = B[i,j] + d_B[i,j]
                Q[i,j] = Q[i,j] + d_Q[i,j]


                # Harvest
                # ----------------------------------------------------------
                # counter for if death rate exceeds growth rate
                if M > Growth:
                    GD_count[i,j] += 1
                else:
                    GD_count[i,j] = 0

                # increment growth/death running averages
                if seed_now[i,j] > 0:
                   Gave[i,j] = Growth # growth [1/timestep]
                   Dave[i,j] = M    # death [1/timestep]
                Gave[i,j] = Gave[i,j] + (Growth-Gave[i,j]) / params[mp_harvest_avg_period]
                Dave[i,j] = Dave[i,j] + (M-Dave[i,j]) / params[mp_harvest_avg_period]

                # check for, and perform harvest
                harv1 = c0
                if params[mp_harvest_schedule] == 0:
                    # fixed harvest
                    if do_harvest[i,j] == 1:
                        if params[mp_harvest_type] == 0:
                            # harvest to seed weight
                            if B[i,j] > c0:
                                harv1 = max(c0,(B[i,j]-params[mp_spp_seed]/params[mp_spp_line_sep])/B[i,j])
                            n_harv[i,j] += 1 # output variable
                            t_harv[i,j] += 1 # internal recording
                        else: # params[mp_harvest_type] == 1:
                            # fractionally harvest, but not if below mp_harvest_kg
                            #if B[i,j] >= params[mp_harvest_kg]*params[mp_spp_line_sep]*1.e3:
                            harv1 = params[mp_harvest_f]
                            n_harv[i,j] += 1 # output variable
                            t_harv[i,j] += 1 # internal recording
                else:
                    #  harvest schedules 1&2
                    if do_harvest[i,j] == 2:
                            # we are done for the season, if not already harvested
                        if params[mp_harvest_type] == 1:  # type 1 - don't final harvest all the way, either perrential (macrocystis) or saving seed weight
                            harv1 = params[mp_harvest_f]
                        else:
                            harv1 = 0.99
                        mask[i,j] = 0
                        n_harv[i,j] += 1
                        t_harv[i,j] += 1
                    elif do_harvest[i,j] == 1:

                        if params[mp_harvest_schedule] == 1:
                            # within final harvest span - check for declining growth
                            #if Gave[i,j]/Dave[i,j] < 1.0:  # test of declining biomass over time
                            if GD_count[i,j] > params[mp_harvest_avg_period]: # test of negative growth over time
                                if params[mp_harvest_type] == 1:  # type 1 - macrocystis, perennial, don't final harvest all the way
                                    harv1 = params[mp_harvest_f]
                                else:
                                    harv1 = 0.99
                                    #harv1 = max(c0,(B[i,j]-params[mp_spp_seed]/params[mp_spp_line_sep])/B[i,j])

                                mask[i,j] = 0
                                n_harv[i,j] += 1
                                t_harv[i,j] += 1

                            # ============================= kg harvest block ==========
                            if harv1 == c0:
                                # conditional harvest up to mp_harvest_nmax - 1; last harvest is final harvest
                                if t_harv[i,j] < (params[mp_harvest_nmax]-1):
                                    # if not already harvested, check to see if conditions are OK for incremental harvest
                                    if B[i,j] > params[mp_harvest_kg]*params[mp_spp_line_sep]*1.e3:
                                        harv1 = params[mp_harvest_f]
                                        n_harv[i,j] += 1
                                        t_harv[i,j] += 1
                            # ============================= end kg harvest block ====
                        else:  #params[mp_harvest_schedule] == 2:
                            # do harvest according to "type"
                            if params[mp_harvest_type] == 0:
                                # harvest to seed weight
                                harv1 = max(c0,(B[i,j]-params[mp_spp_seed]/params[mp_spp_line_sep])/B[i,j])
                                n_harv[i,j] += 1 # output variable
                                t_harv[i,j] += 1 # internal recording
                            elif params[mp_harvest_type] == 1:
                                # fractionally harvest, but not if below mp_harvest_kg
                                if B[i,j] >= params[mp_harvest_kg]*params[mp_spp_line_sep]*1.e3:
                                    harv1 = params[mp_harvest_f]
                                    n_harv[i,j] += 1 # output variable
                                    t_harv[i,j] += 1 # internal recording
                            elif params[mp_harvest_type] == 3:
                                # harvest above target kg
                                if B[i,j] > params[mp_harvest_kg]*1.e3:
                                    harv1 = (B[i,j] - params[mp_harvest_kg]*1.e3)/B[i,j]
                                    n_harv[i,j] += 1 # output variable
                                    t_harv[i,j] += 1 # internal recording




                    #if harv1 > c0:  # reset mean biomass growth/death records so they don't swing around?
                    #   Gave[i,j] = Growth
                    #   Dave[i,j] = M


                harv1 = harv1 * B[i,j]  # biomass harvested
                B[i,j] = B[i,j] - harv1
                harv[i,j] += harv1


                ### sanity checks, at end of calc to prevent output of bad data
                if mask[i,j] > 0:
                    bad_forcing = np.sum(np.isnan(np.array([sst[i,j],par[i,j],NO3[i,j],swh[i,j],
                                                            cmag[i,j],nflux[i,j],B[i,j],Q[i,j]])))
                    if bad_forcing > 0 or sst[i,j] < -3.0 or sst[i,j] > 50.0:
                        mask[i,j] = 0
                        B[i,j] = np.nan
                        Q[i,j] = np.nan
                        harv[i,j] = np.nan
                        d_B[i,j] = np.nan
                        d_Q[i,j] = np.nan
                        GRate[i,j] = np.nan
                        Growth2[i,j] = np.nan
                        d_Be[i,j] = np.nan
                        d_Bm[i,j] = np.nan
                        d_Ns[i,j] = np.nan
                        B_N[i,j] = np.nan
                        n_harv[i,j] = 0
                        Gave[i,j] = np.nan
                        Dave[i,j] = np.nan
                        min_lim[i,j] = 0
                        gQout[i,j] = np.nan
                        gTout[i,j] = np.nan
                        gEout[i,j] = np.nan
                        gHout[i,j] = np.nan




@jit(nopython=True)
def daylength(lat,lon,alt,dte):

    # main function that computes daylength and noon time
    # https://en.wikipedia.org/wiki/Sunrise_equation

    # number of days since Jan 1st, 2000 12:00 UT
    #dte_2000_days = 730490.0
    #n2000 = dte - dte_2000_days + 68.184/86400
    n2000 = dte - 0.5 + 68.184/86400

    # mean solar moon
    Js = n2000 - lon/360.0

    # solar mean anomaly
    M = (357.5291 + 0.98560028*Js)%360.0

    # center
    Mrad = math.radians(M)
    C = 1.9148*math.sin(Mrad) + 0.0200*math.sin(2.0*Mrad) + 0.0003*math.sin(3.0*Mrad)

    # ecliptic longitude
    lambda0 = (M + C + 180.0 + 102.9372)%360.0

    # solar transit
    lambda0_rad = math.radians(lambda0)
    #Jt = 2451545.5 + Js + 0.0053*math.sin(Mrad) - 0.0069*math.sin(2.0*lambda0_rad) # only needed for noontime, not neded

    # Sun declination
    delta_rad = math.asin(math.sin(lambda0_rad)*math.sin(math.radians(23.44)))

    # hour angle (day expressed in geometric degrees)
    h = (math.sin(math.radians(-0.83 - 2.076*math.sqrt(alt)/60.0)) -
         math.sin(math.radians(lat))*math.sin(delta_rad))/(math.cos(math.radians(lat))*math.cos(delta_rad))

    # to avoid meaningless complex angles: forces omega to 0 or 12h
    if (h < -1):
        omega = 180.0
    elif (h > 1):
        omega = 0.0
    else:
        omega = math.degrees(math.acos(h))

    return omega/180.0 # daylength

@jit(nopython=True)
def lambda_NO3(magu,Tw,CD,VmaxNO3,KsNO3,NO3):

        # unit conversions, minima
        magmax = max(1.0,magu*86400.0) # m/day
        Tw_s = max(0.01,Tw)/86400.0 # mean mave period
        NO3_u = NO3*1000.0 # converting from uM to umol/m3

        # some parameters and variables
        n_length = 25
        visc = 1.e-6 * 86400.0
        Dm = (18.0*3.65e-11 + 9.72e-10) * 86400.0
        vval = np.zeros(n_length,np.float32)  # two v's b/c val seems to be a keyword in fortran

        DBL = 10.0 * (visc / (math.sqrt(CD) * abs(magmax)))

        # 1. Oscillatory Flow
        for n in range(1,n_length+1):
            vval[n-1] = (1-math.exp((-Dm * n**2 * math.pi**2 *Tw_s)/(2.0*DBL**2)))/(n**2 * math.pi**2)
        oscillatory = ((4.0*DBL)/Tw_s) * np.sum(vval)

        # 2. Uni-directional Flow
        flow = Dm / DBL
        beta = flow + oscillatory
        return 1.0 + (VmaxNO3 / (beta*KsNO3)) - (NO3_u /KsNO3)


@jit(nopython=True)
def temp_lim(sst,Topt1,K1,Topt2,K2):
    if sst >= Topt1:
        if sst <= Topt2:
            temp_lim = 1.0
        else:
            temp_lim = math.exp(-K2*(sst-Topt2)**2)
    else:
        temp_lim = math.exp(-K1*(sst-Topt1)**2)
    #temp_lim = np.max([0.0,temp_lim]) # correctly returns NAN, but also crashes model currently
    temp_lim = max([0.0,temp_lim]) # returns 0 if temp_lim=NAN !!
    return min(1.0,temp_lim)


if __name__ == '__main__':

    pass