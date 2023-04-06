
import sys,os,pickle,copy

# optional path import for directory containing magpy
sys.path.append(r'/mnt/vat/data/mag/')

import numpy as np
from numpy.lib.nanfunctions import _copyto,_replace_nan
from magpy import mag0
from magpy.mag_species import spp_p_dict,spp_p_dict_harvest
import pandas as pd
import netCDF4
import xarray as xr
import h5py
import matplotlib.pyplot as plt
from multiprocessing import Pool
import xarray as xr


def param_update_no_overwrite(p,new_p):
    '''Like dict.update(), but does not overwrite any keys already present in
    dict.'''
    for k,v in new_p.items():
        if k not in p.keys():
            p[k] = v
    return p


def nansum_old(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    '''Replicates the behavior of numpy.nansum() from older releases, where
    summing axes return np.nan if no valid numbers are found (instead of 0.0
    like in newer numpy versions) '''
    a, mask = _replace_nan(a, 0)
    if mask is None:
        return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    mask = np.all(mask, axis=axis, keepdims=keepdims)
    tot = np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    if np.any(mask):
        tot = _copyto(tot, np.nan, mask)
        #warnings.warn("In Numpy 1.9 the sum along empty slices will be zero.",
        #              FutureWarning)
    return tot


def get_out_slice_nc_open(nc,var_name,time_slice_index=None):
    ''''''
    if var_name in nc.variables.keys():
        if time_slice_index is None:
            arr = nc.variables[var_name][...]
        else:
            arr = nc.variables[var_name][time_slice_index,...]
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        return arr
    else:
        print("%s not found in netCDF4 Dataset"%var_name)
        return None


def get_output_slice(output_file,var_name,time_slice_index=None,None_if_2D=False):
    if output_file.endswith('.mat') or output_file.endswith('.h5'):
        fp = h5py.File(output_file, 'r')
        if 'output' in fp.keys():
            dset = fp['output'][var_name]
        else:
            dset = fp[var_name]
        if time_slice_index is None:
            outdata = dset[...]
        else:
            outdata = dset[time_slice_index,...]
        fp.close()
        return outdata
    else:

        nc = netCDF4.Dataset(output_file,"r")
        if None_if_2D:
            if len(nc.variables[var_name].dimensions) == 2:
                nc.close()
                return None
        outdata = get_out_slice_nc_open(nc,var_name,time_slice_index)
        nc.close()
        return outdata


def open_append_nc(output_file,base_var,metric_var_list,data):
    nc = netCDF4.Dataset(output_file,"a")
    for v in metric_var_list:
        append_nc_var(nc,base_var,v,data[v])
    nc.close()

def append_nc_var(nc,base_var,metric_var,data):
    if metric_var not in nc.variables:
        dims = nc.variables[base_var].dimensions
        if len(np.shape(data)) == 2 and len(dims) == 3: # should be the only case when need to change dims
            dims = dims[1:]
        outVar = nc.createVariable(metric_var, nc.variables[base_var].datatype,
                                   dims, zlib=True)
        outVar.setncatts({'units':nc.variables[base_var].units})
    else:
        outVar = nc.variables[metric_var]
    outVar[...] = data


def create_cwm_annual_dataset(var_list,unit_list,data_list):
    # create dataset
    longitude = np.arange(4320) * 1/12 - 180 + 1/24
    latitude = -1.0*(np.arange(2160) * 1/12 - 90 + 1/24)
    var_dict = {}
    for var,unit,data in zip(var_list,unit_list,data_list):
        var_dict[var] = xr.DataArray(
            dims=['latitude','longitude'],
            coords={'latitude': latitude, 'longitude': longitude},
            attrs={'_FillValue': -999.9,'units': unit}
        )
        if data is not None:
            var_dict[var][...] = data
    return xr.Dataset(var_dict)

def create_write_cwm_annual_dataset(var_list,unit_list,data_list,outfile,
                                    dtype_list=None,mode='w'):
    ds = create_cwm_annual_dataset(var_list,unit_list,data_list)
    #for v,d in zip(var_list,data_list):
    #    ds[v][...] = d
    my_dtype_list = [np.float32]*len(var_list) if dtype_list is None else dtype_list
    encoding = {v:{"dtype":dtp,"zlib": True, "complevel": 4} for v,dtp in zip(var_list,my_dtype_list)}
    ds.to_netcdf(outfile,mode=mode,format='netCDF4',encoding=encoding)


def nc_var_exists(output_file,var_name):
    with netCDF4.Dataset(output_file) as nc:
        if var_name in var_name in nc.variables.keys():
            return True
        else:
            return False

def output_dtype(varname):
    if 'n_harv' in varname:
        return np.int32
    else:
        return np.float32



#def copy_nc_annual_vars(nc_in,nc_out):





def add_time_metric_to_nc_xr(output_file,var_name,metrics=['sum','mean'],pre_calc=None):

    if pre_calc is not None:
        vnew_list = [var_name + '_' + m for m in metrics]
        units_list = ['same as unsummed' for v in vnew_list]
        # pre_calc should already be a list of data arrays for writing
        dtypes = [output_dtype(v) for v in vnew_list]
        print('Writing to output nc:',vnew_list)
        create_write_cwm_annual_dataset(vnew_list,units_list,pre_calc,output_file,
                                        dtype_list=dtypes,mode='a')
    else:
        time_data = get_output_slice(output_file,var_name,None_if_2D=True)
        if time_data is not None:
            data = {}
            for m in metrics:
                if m == 'sum':
                    data[var_name + '_' + m] = nansum_old(time_data,axis=0)
                elif m == 'max':
                    data[var_name + '_' + m] = np.nanmax(time_data,axis=0)
                elif m == 'mean':
                    data[var_name + '_' + m] = np.nanmean(time_data,axis=0)
                elif m == 'min':
                    data[var_name + '_' + m] = np.nanmin(time_data,axis=0)
                elif m == 'median':
                    data[var_name + '_' + m] = np.nanmedian(time_data,axis=0)
            vnew_list = data.keys()
            units_list = ['same as unsummed' for v in vnew_list]
            data_list = [data[v] for v in vnew_list]
            dtypes = [output_dtype(v) for v in vnew_list]
            print('Writing to output nc:',vnew_list)
            create_write_cwm_annual_dataset(vnew_list,units_list,data_list,output_file,
                                            dtype_list=dtypes,mode='a')

def add_time_metric_to_nc(output_file,var_name,metrics=['sum','mean'],sanity_mask=None):

    time_data = get_output_slice(output_file,var_name,None_if_2D=True)
    if time_data is not None:
        data = {}
        for m in metrics:
            var_metric = var_name + '_' + m
            if m == 'sum':
                data[var_metric] = nansum_old(time_data,axis=0)
            elif m == 'max':
                data[var_metric] = np.nanmax(time_data,axis=0)
            elif m == 'mean':
                data[var_metric] = np.nanmean(time_data,axis=0)
            elif m == 'min':
                data[var_metric] = np.nanmin(time_data,axis=0)
            elif m == 'median':
                data[var_metric] = np.nanmedian(time_data,axis=0)
            if sanity_mask is not None:
                data[var_metric] = sanitize_results(data[var_metric], sanity_mask)

        vnew_list = data.keys()
        print('Writing to output nc:',vnew_list)
        open_append_nc(output_file,var_name,vnew_list,data)


def generate_standard_runs(output_path,seed_path,base_params,forcing_var_meta,
                  std_param_overides={'Eucheuma':{},
                                      'Saccharina':{},
                                      'Macrocystis':{},
                                      'Porphyra':{},
                                      'Sargassum':{}},seed_ext='_modef9'):
    run_sets = []
    for spp in ['Eucheuma','Saccharina','Macrocystis','Sargassum','Porphyra',]:
        my_params = copy.deepcopy(base_params)
        my_params.update(std_param_overides[spp])
        my_params = param_update_no_overwrite(my_params,spp_p_dict_harvest[spp])

        run_name_1 = my_params['run_name']
        for flux_lim in [0,1]:
            my_params['mp_N_flux_limit'] = flux_lim
            my_params['run_name'] = run_name_1 + '_' + spp + '_f%i'%flux_lim
            seed_name_stub = seed_paper_name_stub(my_params)
            forcing_var_meta['seed']['path'] = seed_path
            forcing_var_meta['seed']['fname'] = seed_name_stub + '_f0' + '_seed_month_multiyear'+seed_ext+'.nc'

            run_sets.append((copy.deepcopy(my_params),copy.deepcopy(forcing_var_meta)))
            #mag0.MAG0(mag0.build_run_params(my_params),forcing_var_meta=forcing_var_meta).compute()
    return run_sets


def write_hist_fig(v,data,rundir,max_not_mean=True,kind='p90'):
    if len(np.shape(data)) > 2:
        if max_not_mean:
            my_data = np.nanmax(data,axis=0)
        else:
            my_data = np.nansum(data,axis=0)
    else:
        my_data = data

    my_data = my_data.ravel()

    plt.hist(data[np.isfinite(data)],bins=50)
    if v == 'harv':
        plt.title('Mean Harvest')
        plt.xlabel('g/m^2')
    elif v == 'n_harv':
        plt.title('Number Harvests')
        plt.xlabel('#')
    elif v == 'Growth2':
        plt.title('Growth')
        plt.xlabel('g/m^2')
    elif v == 'B':
        plt.title('Biomass')
        plt.xlabel('g/m^2')

    plt.savefig(os.path.join(rundir,'hist_%s_%s.png'%(v,kind)),dpi=600,bbox_inches='tight')
    plt.close()
    plt.clf()

def write_hist_fig_v2(v,data,rundir):

    plt.hist(data[np.isfinite(data)],bins=50)
    if v.startswith('harv'):
        plt.title('Mean Harvest')
        plt.xlabel('g/m^2')
    elif v.startswith('n_harv'):
        plt.title('Number Harvests')
        plt.xlabel('#')
    elif v.startswith('Growth2'):
        plt.title('Growth')
        plt.xlabel('g/m^2')
    elif v.startswith('B'):
        plt.title('Biomass')
        plt.xlabel('g/m^2')
    elif v.startswith('d_Be'):
        plt.title('Exudation')
        plt.xlabel('mg N / m2')
    elif v.startswith('d_Bm'):
        plt.title('Dead Biomass')
        plt.xlabel('g DW / m2')
    elif v.startswith('d_Ns'):
        plt.title('Total Nutrient Uptake')
        plt.xlabel('mg N / g DW')    # is this right?

    plt.savefig(os.path.join(rundir,'hist_%s.png'%v),dpi=600,bbox_inches='tight')
    plt.close()
    plt.clf()

def output_sums(run_dir,area,mask_shelf,mask_coast,vars_to_sum=['harv','n_harv','B'],write_hist=False):
    path,run_name = os.path.split(run_dir)
    pkl_fname = os.path.join(run_dir,"mag0_params_"+run_name+".pkl")
    pkl_pp_fname = os.path.join(run_dir,run_name+"_postprocess.pkl")
    csv_pp_fname = os.path.join(run_dir,run_name+"_postprocess.csv")
    nc_fname = os.path.join(run_dir,"mag0_output_"+run_name+".nc")

    print('Summing run: %s'%run_name)

    with open(pkl_fname,'rb') as fp:
        run_sum = pickle.load(fp)

    nc = netCDF4.Dataset(nc_fname,'r')

    harv = get_out_slice_nc_open(nc,'harv_sum')
    if harv is None:
        harv = get_out_slice_nc_open(nc,'harv')

    if len(np.shape(harv)) > 2:
        harv = np.nansum(harv,axis=0)
    hm = np.nanquantile(harv[harv>0],0.9)
    mask = harv >= hm
    outdata = harv*area*1.0e6/1.0e15
    run_sum['harv_sum_90'] = np.nansum(outdata[mask])
    run_sum['harv_sum'] = np.nansum(outdata)
    if write_hist:
        harv_p90 = harv[mask]
        write_hist_fig('harv',harv_p90[harv_p90 >= 150.],run_dir,kind='p90')
        harv_shelf = harv[mask_shelf]
        write_hist_fig('harv',harv_shelf[harv_shelf >= 150.],run_dir,kind='shelf')
        harv_coast = harv[mask_coast]
        write_hist_fig('harv',harv_coast[harv_coast >= 150.],run_dir,kind='coast')
    print('   summed: harv')

    #outdata = get_out_slice_nc_open(nc,'Growth2')
    #if len(np.shape(outdata)) > 2:
    #    outdata = np.nansum(outdata,axis=0)
    #outdata = outdata*area*1.0e6/1.0e15
    #run_sum['Growth2_sum_90'] = np.nansum(outdata[mask])
    #print('   summed: Growth2')

    outdata = get_out_slice_nc_open(nc,'B_max')
    if outdata is None:
        outdata = get_out_slice_nc_open(nc,'B')

    if outdata is not None:
        if len(np.shape(outdata)) > 2:
            outdata = np.nanmax(outdata,axis=0)
        run_sum['B_max_90'] = np.nansum(outdata[mask])
        run_sum['B_mean_90'] = np.nanmean(outdata[mask])
        run_sum['B_median_90'] = np.nanmedian(outdata[mask])
        run_sum['B_max'] = np.nansum(outdata)
        run_sum['B_mean'] = np.nanmean(outdata)
        run_sum['B_median'] = np.nanmedian(outdata)
        print('   summed: B')
        if write_hist:
            hist_data = outdata[mask]
            write_hist_fig('B',hist_data[harv_p90 >= 150.],run_dir,kind='p90')
            hist_data = outdata[mask_shelf]
            write_hist_fig('B',hist_data[harv_shelf >= 150.],run_dir,run_dir,kind='shelf')
            hist_data = outdata[mask_coast]
            write_hist_fig('B',hist_data[harv_coast >= 150.],run_dir,kind='coast')

    n_harv = get_out_slice_nc_open(nc,'n_harv_sum')
    if n_harv is None:
        n_harv = get_out_slice_nc_open(nc,'n_harv')
    if len(np.shape(n_harv)) > 2:
        n_harv = np.nansum(n_harv,axis=0)
    n_harv = n_harv.astype(np.float32)
    if write_hist:
        hist_data = n_harv[mask]
        write_hist_fig('n_harv',hist_data[harv_p90 >= 150.],run_dir,kind='p90')
        hist_data = n_harv[mask_shelf]
        write_hist_fig('n_harv',hist_data[harv_shelf >= 150.],run_dir,run_dir,kind='shelf')
        hist_data = n_harv[mask_coast]
        write_hist_fig('n_harv',hist_data[harv_coast >= 150.],run_dir,kind='coast')

    run_sum['n_harv_sum_90'] = np.nansum(n_harv[mask])
    run_sum['n_harv_mean_90'] = np.nanmean(n_harv[mask])
    bm_per_h = harv / n_harv
    run_sum['harv_mean_90'] = np.nanmean(harv[mask])
    run_sum['harv_p_harv_mean_90'] = np.nanmean(bm_per_h[mask])

    run_sum['n_harv_sum'] = np.nansum(n_harv)
    run_sum['n_harv_mean'] = np.nanmean(n_harv)
    run_sum['harv_mean'] = np.nanmean(harv)
    run_sum['harv_p_harv_mean'] = np.nanmean(bm_per_h)

    print('   summed: n_harv')

    with open(pkl_pp_fname,'wb') as fp:
        pickle.dump(run_sum,fp)
    pd.DataFrame([run_sum]).to_csv(csv_pp_fname)

    return run_sum


def get_output_stats(run_dir,dataframe=False):
    path,run_name = os.path.split(run_dir)
    pkl_pp_fname = os.path.join(run_dir,run_name+"_postprocess.pkl")
    csv_pp_fname = os.path.join(run_dir,run_name+"_postprocess.csv")
    if dataframe:
        if os.path_exists(csv_pp_fname):
            return pd.read_csv(csv_pp_fname)
        else:
            return None
    else:
        if os.path_exists(pkl_pp_fname):
            with open(pkl_pp_fname,'rb') as fp:
                run_sum = pickle.load(fp)
            return run_sum
        else:
            return None


def sanitize_results(results,sanity_mask):
    if results.dtype=='float32' or results.dtype=='float64':
        results[sanity_mask] = np.nan
    else:
        results[sanity_mask] = 0
    return results


def output_stats(run_dir,area,mask_dict,stats_dict,
                 quantiles,quantile_var='harv',quantile_var_min=150.0,
                 write_hist=False,check=True,mode='a',sanity_mask=None,specified_nc=None):
    """
    Parameters
    ----------
    run_dir : TYPE
        DESCRIPTION.
    area : TYPE
        DESCRIPTION.
    mask_shelf : TYPE
        DESCRIPTION.
    mask_coast : TYPE
        DESCRIPTION.
    stats_dict : TYPE
        DESCRIPTION.
    quantiles : TYPE
        DESCRIPTION.
    quantile_var : TYPE, optional
        DESCRIPTION. The default is 'harv'.
    write_hist : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    run_sum : TYPE
        DESCRIPTION.

    """
    if specified_nc is not None:
        path,run_name = os.path.split(specified_nc)
        pkl_fname = None
        pkl_pp_fname = specified_nc[:-3] + "_postprocess.pkl"
        csv_pp_fname = specified_nc[:-3] +"_postprocess.csv"
        nc_fname = specified_nc
        nc_supl = nc_fname[:-3] + '_supl1.nc'
        run_sum = {'nc_file':specified_nc}  # we don't have any run info with just the nc file
    else:
        path,run_name = os.path.split(run_dir)
        pkl_pp_fname = os.path.join(run_dir,run_name+"_postprocess.pkl")
        pkl_fname = os.path.join(run_dir,"mag0_params_"+run_name+".pkl")
        csv_pp_fname = os.path.join(run_dir,run_name+"_postprocess.csv")
        nc_fname = os.path.join(run_dir,"mag0_output_"+run_name+".nc")
        nc_supl = nc_fname[:-3] + '_supl1.nc'

    if check and os.path.exists(pkl_pp_fname):
        print('Already postprocessed:',run_name)
    else:
        print('Generating stats: %s'%run_name)

        if os.path.exists(pkl_pp_fname) and mode=='a':
            with open(pkl_pp_fname,'rb') as fp:
                run_sum = pickle.load(fp)
        else:
            if pkl_fname is not None:
                with open(pkl_fname,'rb') as fp:
                    run_sum = pickle.load(fp)

        nc = netCDF4.Dataset(nc_fname,'r')
        nc2 = None
        if os.path.exists(nc_supl):
            nc2 = netCDF4.Dataset(nc_supl,'r')

        q_mask = {}
        for quant in quantiles:
            qvar = get_out_slice_nc_open(nc,quantile_var)

            if len(np.shape(qvar)) > 2:
                qvar = np.nansum(qvar,axis=0)
            if sanity_mask is not None:
                qvar = sanitize_results(qvar, sanity_mask)

            if quant == 'all':
                q_mask['all'] = qvar>quantile_var_min
            else:
                qvar_q = np.nanquantile(qvar[qvar>quantile_var_min],quant)
                q_mask['%4.2f'%quant] = qvar >= qvar_q

        for v,stats in stats_dict.items():
            if v == 'netprod' and nc2 is not None:
                outdata = get_out_slice_nc_open(nc2,v)
            else:
                outdata = get_out_slice_nc_open(nc,v)

            if len(np.shape(outdata)) > 2:
                outdata = np.nansum(outdata,axis=0)
            if sanity_mask is not None:
                outdata = sanitize_results(outdata, sanity_mask)


            if 'sum' in stats:
                outsum = outdata*area
            for mname,mm in mask_dict.items():
                for qname,qm in q_mask.items():
                    qm_mask = np.logical_and(qm,mm)
                    for s in stats:
                        sname = '_'.join([v,s,mname,qname])
                        if s == 'sum':
                            outsum = outdata*area
                            run_sum[sname] = np.nansum(outsum[qm_mask])
                        elif s == 'mean':
                            run_sum[sname] = np.nanmean(outdata[qm_mask])
                        elif s == 'median':
                            run_sum[sname] = np.nanmedian(outdata[qm_mask])
                        elif s == 'max':
                            run_sum[sname] = np.nanmax(outdata[qm_mask])
                        elif s == 'min':
                            run_sum[sname] = np.nanmin(outdata[qm_mask])
                        print('    Stats:',sname)

                if write_hist and specified_nc is not None:
                    write_hist_fig_v2('_'.join([v,mname,qname]),outdata[qm_mask],run_dir)

        nc.close()
        if nc2 is not None:
            nc2.close()
        with open(pkl_pp_fname,'wb') as fp:
            pickle.dump(run_sum,fp)
        pd.DataFrame([run_sum]).to_csv(csv_pp_fname)


def temporary_sum_function(nc,var_name,run_sum,mask,str_mask,metrics):
    data = get_out_slice_nc_open(nc,var_name)
    if data is not None:
        for m in metrics:
            if m == 'sum':
                run_sum[var_name+'_'+m+'_'+str_mask] = nansum_old(data[mask])
            elif m == 'max':
                run_sum[var_name+'_'+m+'_'+str_mask] = np.nanmax(data[mask])
            elif m == 'mean':
                run_sum[var_name+'_'+m+'_'+str_mask] = np.nanmean(data[mask])
            elif m == 'min':
                run_sum[var_name+'_'+m+'_'+str_mask] = np.nanmin(data[mask])
            elif m == 'median':
                run_sum[var_name+'_'+m+'_'+str_mask] = np.nanmedian(data[mask])
    return run_sum


def get_mask(mask_fn, mask_name):
    """Some options: coast_99, coast_54, coast_27, coast_18, coast_9, shelf_mask_100, shelf_mask_150,
    shelf_mask_200"""
    if mask_fn.endswith('.h5') or mask_fn.endswith('.mat'):
        with h5py.File(mask_fn, "r") as fp:
            amask = fp["analysis_masks"][mask_name][...]
        return np.transpose(amask).astype(bool)
    elif mask_fn.endswith('.nc'):
        ds=xr.open_dataset(mask_fn)
        amask = ds[mask_name][...].values
        ds.close()
        return amask

def get_area(cwm_grid_matfile,area_varname = "area"):
    with h5py.File(cwm_grid_matfile, 'r') as fp:
        amask = fp['analysis_masks'][area_varname][...]
    return np.transpose(amask)



def eez_mask(eez_nc_fname,name='eez'):
    nc = netCDF4.Dataset(eez_nc_fname,mode="r")
    eez_mask = nc[name][...].astype(np.bool_)
    if np.ma.isMaskedArray(eez_mask):
        eez_mask = eez_mask.filled(False)
    nc.close()
    return eez_mask

def eez_usa(eez_nc_fname):
    masks = {'eez_lower48': eez_mask(r'X:/data/mag/eez_masks.nc', 'United States Exclusive Economic Zone'),
             'eez_ak': eez_mask(r'X:/data/mag/eez_masks.nc', 'United States Exclusive Economic Zone --Alaska--'),
             'eez_hi': eez_mask(r'X:/data/mag/eez_masks.nc', 'United States Exclusive Economic Zone --Hawaii--')}
    masks['eez_usa'] = np.logical_or(masks['eez_lower48'], masks['eez_ak'])
    masks['eez_usa'] = np.logical_or(masks['eez_usa'], masks['eez_hi'])
    return masks


def seed_paper_name_stub(p,flux_lim=None):
    run_name = '%s_s%i_t%i_freq%i_sp%i_nh%i_frac%3.1f_kg%4.2f_kcap%i'%(p['spp'],
                                                    p['mp_harvest_schedule'],
                                                    p['mp_harvest_type'],
                                                    p['mp_harvest_freq'],
                                                    p['mp_harvest_span'],
                                                    p['mp_harvest_nmax'],
                                                    p['mp_harvest_f'],
                                                    p['mp_harvest_kg'],
                                                    p['mp_spp_kcap'])
    if flux_lim is not None:
        run_name += '_f%i'%flux_lim
    return run_name


def pickles_to_csv(run_generator,csv_fname,postprocessed=True):
    run_sums = []
    for run_dir in run_generator:
        path,run_name = os.path.split(run_dir)
        if postprocessed:
            pkl_fname = os.path.join(run_dir,run_name+"_postprocess.pkl")
        else:
            pkl_fname = os.path.join(run_dir,"mag0_params_"+run_name+".pkl")
        with open(pkl_fname,'rb') as fp:
            run_sums.append(pickle.load(fp))
    pd.DataFrame(run_sums).to_csv(csv_fname)


def postprocess_old(run_dir, area_path, mask_matfile,
                sums=True, histograms=True, sum_maps=True,
                sum_vars = ['Growth2','d_Be','d_Bm','d_Ns','harv','n_harv'],
                non_sum_vars = ['B','Grate'],
                sum_var_metrics = ['sum'],
                non_sum_var_metrics = ['max']):


    mask_shelf = get_mask(mask_matfile,'shelf_mask_100')
    mask_coast = get_mask(mask_matfile,'coast_18')
    area = np.loadtxt(area_path)

    path,run_name = os.path.split(run_dir)
    nc_fname = os.path.join(run_dir,"mag0_output_"+run_name+".nc")

    if sum_maps:
        for v in sum_vars:
            add_time_metric_to_nc(nc_fname,v,metrics=sum_var_metrics)
        for v in non_sum_vars:
            add_time_metric_to_nc(nc_fname,v,metrics=non_sum_var_metrics)

    if sums:
        output_sums(run_dir,area,mask_shelf,mask_coast,write_hist=histograms)


def add_net_production(nc_output,check,sanity_mask=None):
    if os.path.exists(nc_output):
        base_nc,ext = os.path.splitext(nc_output)
        suppl_nc = base_nc + '_supl1.nc'
        if not (check and os.path.exists(suppl_nc)):
            growth = get_output_slice(nc_output,'Growth2')
            if len(np.shape(growth)) > 2:
                growth = np.nansum(growth,axis=0)
            death = get_output_slice(nc_output,'d_Bm')
            if len(np.shape(death)) > 2:
                death = np.nansum(death,axis=0)
            print('Writing to output nc:','netprod')
            netp = growth-death
            if sanity_mask is not None:
                netp[sanity_mask] = np.nan
            #try:
            #    open_append_nc(nc_output,'Growth2',['netprod'],{'netprod':growth-death})
            #except:
            #    print('Cannot append to results nc .. trying supplemental nc ...')
            create_write_cwm_annual_dataset(['netprod'],['g C m-2'],[netp],suppl_nc,
                                        dtype_list=None,mode='w')

# this mask is a postive mask of bad values

def generate_sanity_mask(output_file,var_names,max_values):
    mask = None
    for v,mv in zip(var_names,max_values):
        time_data = get_output_slice(output_file,v,None_if_2D=True)
        if time_data is not None:
            values = nansum_old(time_data,axis=0)
        else:
            values = get_output_slice(output_file,v,None_if_2D=False)
        if mask is None:
            mask = values >= mv
        else:
            mask = np.logical_or(mask,values >= mv)
    return mask


def postprocess(run_dirs,
                area,
                mask_dict = {'world':np.s_[...]},
                stats_dict = {v:['sum','mean','median'] for v in ['Growth2',
                                                                  'd_Be',
                                                                  'd_Bm',
                                                                  'd_Ns',
                                                                  'harv',
                                                                  'n_harv']},
                quantiles = ['all',0.9],
                quantile_var = 'harv',
                quantile_var_min = 150.0,
                sums=True,
                histograms=False,
                sum_maps=False,
                check=False,
                mode='a',
                netp = True,
                timeseries_points=[],
                sanity_check_max=[['Growth2','harv'],[1.0e5,1.0e5]]):


    for r in run_dirs:
        path,run_name = os.path.split(r)
        print("Postprocessing:",run_name)
        nc_fname = os.path.join(r,"mag0_output_"+run_name+".nc")

        sanity_mask = None
        if sanity_check_max is not None:
            sanity_check_vars = []
            for v in sanity_check_max[0]:
                if v in stats_dict.keys():
                    sanity_check_vars.append(v)
            if sanity_check_vars:
                sanity_mask = generate_sanity_mask(nc_fname,sanity_check_vars,sanity_check_max[1])

        if os.path.exists(nc_fname):
            if os.path.getsize(nc_fname) > 8128000:
            # try:
                if netp:
                    add_net_production(nc_fname,check,sanity_mask=sanity_mask)

                if sum_maps:
                    for v,metrics in stats_dict.items():
                        add_time_metric_to_nc(nc_fname,v,metrics=metrics,sanity_mask=sanity_mask)

                if sums:
                    output_stats(r,area,mask_dict,stats_dict,
                             quantiles,quantile_var,quantile_var_min,
                             histograms,check,mode,sanity_mask=sanity_mask)

                if timeseries_points:
                    pkl_ts_fname = os.path.join(r,run_name+"_timeseries.pkl")
                    if not (mode=='a' and os.path.exists(pkl_ts_fname)):
                        print('Pickling time series...')
                        ds_in = xr.open_dataset(nc_fname,decode_times=False)
                        time_series={}
                        for name,tsvars,ij,ll in timeseries_points:
                            if name=='S_Mississippi':
                                dude=1
                            time_series[name] = {}
                            time_series[name]['latitude'] = ll[0]
                            time_series[name]['longitude'] = ll[1]
                            for v in tsvars:
                                if len(ds_in[v].dims) > 2:
                                    time_series[name][v] = ds_in[v][:,ij[0],ij[1]].values
                                else:
                                    time_series[name][v] = ds_in[v][ij[0],ij[1]].values

                        with open(pkl_ts_fname,'wb') as fp:
                            pickle.dump(time_series,fp)
            # except:
            #     print("Error postprocessing:",run_name,nc_fname)

def csv_from_pp(run_dirs_or_plk_files,out_csv):
    # run again over simulation sums and join
    run_dicts = []
    for i,mc in enumerate(run_dirs_or_plk_files):
        if mc.endswith("_postprocess.pkl"):
            pkl_pp_fname = mc
        else:
            path,run_name = os.path.split(mc)
            pkl_pp_fname = os.path.join(mc,run_name+"_postprocess.pkl")
        if os.path.exists(pkl_pp_fname):
            try:
                with open(pkl_pp_fname,'rb') as fp:
                    run_dicts.append(pickle.load(fp))
            except:
                print(mc,'summary corrupt!')
        else:
            print(mc,'summary missing!')
    df = pd.DataFrame(run_dicts)
    df.to_csv(out_csv)


def ts_analysis(mc_dirs,out_nc):
    '''Produce NetCDF files of timeseries outputs'''
    # run again over simulation sums and join
    run_dicts = []
    tsa = None
    for i,mc in enumerate(mc_dirs):
        path,run_name = os.path.split(mc)
        pkl_pp_fname = os.path.join(mc,run_name+"_timeseries.pkl")
        if os.path.exists(pkl_pp_fname):
            try:
                with open(pkl_pp_fname,'rb') as fp:
                    ts = pickle.load(fp)
            except:
                print(mc,'summary corrupt!')
        else:
            print(mc,'summary missing!')
        if tsa is None:
            tsa={}
            for loc in ts.keys():
                tsa[loc]={}
                tsa['run_name'] = []
        tsa['run_name'].append(run_name)
        for loc in ts.keys():
            for v,data in ts[loc].items():
                if not v in tsa[loc].keys():
                    tsa[loc][v] = []
                tsa[loc][v].append(data)

    ds = xr.Dataset()
    ds['run_name'] = ('runs',),tsa['run_name']
    #ds['longitude'] = ('runs',),tsa['longitude']
    #ds['latitude'] = ('runs',),tsa['latitude']
    names = list(tsa.keys())
    names.remove('run_name')
    #del names['latitude']
    #del names['longitude']

    for loc in names:
        for v,data in tsa[loc].items():
            vname = '__'.join([loc,v])
            data_arr = np.asarray(data)
            shp = np.shape(data_arr)
            if len(shp) > 1:
                ds[vname] = ('runs','time_%i'%shp[-1]),data_arr
            else:
                ds[vname] = ('runs',),data_arr
    ds.to_netcdf(out_nc,format='netCDF4')

def ij_from_latlon(lat,lon):
    '''Get i,j indices to the G-MACMODS 1/12 degree grid from lat,lon'''
    lon360 = 360.+lon if lon<0 else lon
    longitude = np.arange(4320) * 1 / 12 - 180 + 1 / 24
    longitude[longitude<0.] = longitude[longitude<0.] + 360.
    latitude = -1.0 * (np.arange(2160) * 1 / 12 - 90 + 1 / 24)
    return np.argmin(np.abs(latitude-lat)),np.argmin(np.abs(longitude-lon360))


def postprocess_threaded_to_csv(output_path_or_list,nthreads,out_csv,pp_args,debug=False):
    '''Run postprocessing for G-MACMODS in a parallel fashion'''
    if type(output_path_or_list) != list:
        mc_dirs = [ name for name in os.listdir(output_path_or_list) if os.path.isdir(os.path.join(output_path_or_list, name)) ]
    else:
        mc_dirs = output_path_or_list

    #divide dirs into semi-equal lists
    n_dirs= len(mc_dirs)
    nx_thread = int(np.ceil(n_dirs/nthreads))
    pool = Pool(nthreads)
    for i in range(nthreads):
        sl = i*nx_thread
        el = min(sl+nx_thread,n_dirs)
        if not debug:
            pool.apply_async(postprocess,args=[mc_dirs[sl:el]]+pp_args)
        else:
            if len(pp_args) == 13:  # legacy useage
                postprocess(mc_dirs[sl:el],pp_args[0],pp_args[1],pp_args[2],pp_args[3],
                            pp_args[4],pp_args[5],pp_args[6],pp_args[7],pp_args[8],
                            pp_args[9],pp_args[10],pp_args[11],pp_args[12])
            else:
                postprocess(mc_dirs[sl:el],pp_args[0],pp_args[1],pp_args[2],pp_args[3],
                            pp_args[4],pp_args[5],pp_args[6],pp_args[7],pp_args[8],
                            pp_args[9],pp_args[10],pp_args[11],pp_args[12],pp_args[13],pp_args[14])
    pool.close()
    pool.join()
    del pool

    csv_from_pp(mc_dirs,out_csv)
    #ts_analysis(mc_dirs,out_csv[:-4]+'_timeseries.nc')


