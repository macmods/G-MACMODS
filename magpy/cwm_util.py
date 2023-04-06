import os, datetime, csv, math, copy, shutil
import numpy as np
import pandas as pd
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import cartopy
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, date2num

#from mpl_toolkits.basemap import Basemap
# from pyhdf.SD import SD, SDC
from numpy.lib.nanfunctions import _copyto, _replace_nan
import xarray as xr
import netCDF4
from scipy.interpolate import griddata
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
from scipy.interpolate import RegularGridInterpolator
from scipy import spatial
import h5py
from numba import jit, njit
import numba
#import shapely.geometry as sgeom

# from fast_interp import interp1d

# test fast_interp to compile numba functions
# random_noise_size = 0.0
# a = 0.5
# b = 1.0
# x, h = np.linspace(a, b, n, endpoint=True, retstep=True)
# test_x = (x + np.random.rand(*x.shape) * h)[:-1]
# def test_function(x):
#     return np.exp(x) * np.cos(x) + x ** 2 / (1 + x)
# f = test_function(x) + 2 * (np.random.rand(*x.shape) - 0.5) * random_noise_size
# fa = test_function(test_x)
# interpolater = interp1d(a, b, h, f, k=k)
# fe = interpolater(test_x)
# ----------------------------------------------------------------


# from stompy.spatial.interpXYZ import interpXYZ

if os.name == 'nt': # test for windows
    default_cwm_mask = r'X:\data\CWM\regions_and_masks\cwm_mask_20220412_from_hycom.txt'
    default_cwm_grid_area = r"X:\data\CWM\grid\area_twelfth_degree.txt"
    
else:
    default_cwm_mask = r'/mnt/vat/data/CWM/regions_and_masks/cwm_mask_20220412_from_hycom.txt'
    default_cwm_grid_area = r"/mnt/vat/data/CWM/grid/area_twelfth_degree.txt"


monthly_day_mid = [
    15.5,
    31 + 14,
    59 + 15.5,
    90 + 15,
    120 + 15.5,
    151 + 15,
    181 + 15.5,
    212 + 15.5,
    243 + 15,
    273 + 15.5,
    304 + 15,
    334 + 15.5,
]
monthly_day_start = [1,
    32,
    60,
    91,
    121,
    152,
    182,
    213,
    244,
    274,
    305,
    335]


def cwm_map(cwm_data, clims, title, cmap, figsize=[9, 4]):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    img = plt.imshow(cwm_data, cmap=cmap, clim=clims)
    cbar = plt.colorbar(img)
    # plt.grid(linestyle='--',linewidth=0.25)
    # plt.axis('off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    plt.title(title)
    print("Creating: %s" % title)
    return fig, cbar


def cwm_map_mercator(cwm_data, clims, title, ylatb, xlonb, cmap, figsize=[9, 4]):
    fig = plt.figure(figsize=figsize)
    bmap = Basemap(
        projection="mercator", resolution="l", lon_0=0.0, lat_0=0.0
    )  # h=10km resolution, for coasts, etc
    bmap.pcolor(
        xlonb,
        ylatb,
        cwm_data,
        cmap=cmap,
        vmin=clims[0],
        vmax=clims[1],
        latlon=True,
        shading="auto",
    )
    bmap.colorbar(location="bottom", format="%.1f")
    bmap.drawcoastlines(linewidth=0.25)
    plt.title(title)
    print("Creating: %s" % title)
    return bmap


def cwm_map_robin_old(cwm_data, clims, title, ylatb, xlonb, cmap):
    """maybe for pubs but takes a long time to plot."""
    fig = plt.figure(figsize=[8, 6])
    bmap = Basemap(
        projection="robin", resolution="l", lon_0=0.0, lat_0=0.0
    )  # h=10km resolution, for coasts, etc
    bmap.pcolormesh(
        xlonb,
        ylatb,
        cwm_data,
        cmap=cmap,
        vmin=clims[0],
        vmax=clims[1],
        latlon=True,
        shading="auto",
    )
    cbar = bmap.colorbar(location="bottom", format="%.1f")
    bmap.drawcoastlines(linewidth=0.25)
    plt.title(title)
    return bmap, cbar


def cwm_map_lambert_USA(cwm_data, clims, title, ylatb, xlonb, cmap, mask=None):
    fig = plt.figure(figsize=[10, 6])

    ax = fig.add_subplot(1,1,1,projection=ccrs.Mercator())
    #ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.Mercator())
    ax.set_extent([-130, -64.5, 20, 50], ccrs.Geodetic())
    if mask is not None:
        cwm_data[mask] = np.nan
    map_data = ax.pcolormesh(
        xlonb,
        ylatb,
        cwm_data,
        cmap=cmap,
        vmin=clims[0],
        vmax=clims[1],
        shading="flat",
        transform=ccrs.PlateCarree()
    )
    ax.coastlines()
    ax.add_feature(cfeature.LAND,zorder=2)
    plt.title(title)
    cbar = plt.colorbar(map_data,location="right", format="%.1f")

    # Alaska
    add_insetmap((0.15, 0.4, 0.4, 0.4),(-195+360, -132+360, 46, 73), cwm_data, clims, "Alaska", ylatb, xlonb, cmap)

    # add inset Hawaii
    add_insetmap((0.5, 0.55, 0.15, 0.15), (-185, -150, 15, 33), cwm_data, clims, "Hawai'i", ylatb, xlonb, cmap)

    return fig, ax, cbar


def add_insetmap(axes_extent, map_extent, cwm_data, clims, title, ylatb, xlonb, cmap):
    # create new axes, set its projection
    use_projection = ccrs.Mercator(central_longitude=180)     # preserve shape well
    #use_projection = ccrs.PlateCarree()   # large distortion in E-W for Alaska
    geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
    sub_ax = plt.axes(axes_extent, projection=use_projection)  # normal units
    sub_ax.set_extent(map_extent, geodetic)  # map extents
    map_data = sub_ax.pcolormesh(
        xlonb,
        ylatb,
        cwm_data,
        cmap=cmap,
        vmin=clims[0],
        vmax=clims[1],
        shading="flat",
        transform=ccrs.PlateCarree()
    )
    sub_ax.coastlines()
    sub_ax.add_feature(cfeature.LAND,zorder=2)
    sub_ax.set_title(title)

    # plot box around the map
    extent_box = sgeom.box(map_extent[0], map_extent[2], map_extent[1], map_extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), color='none', linewidth=0.05,zorder=3)


def cwm_map_robin(cwm_data, clims, title, ylatb, xlonb, cmap):
    # assumes cartopy is already imported
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(1,1,1,projection=ccrs.Robinson())
    ax.set_global()
    #ax.contourf(lon, lat, data)

    map_data = ax.pcolormesh(
        xlonb,
        ylatb,
        cwm_data,
        cmap=cmap,
        vmin=clims[0],
        vmax=clims[1],
        shading="flat",
        transform=ccrs.PlateCarree()
    )
    cbar = plt.colorbar(map_data,location="bottom", format="%.1f")
    ax.coastlines()
    ax.add_feature(cfeature.LAND,zorder=4)
    plt.title(title)
    return fig, ax, cbar


def open_append_nc(output_file, base_var, metric_var_list, data):
    with netCDF4.Dataset(output_file, "a") as nc:
        for v in metric_var_list:
            append_nc_var(nc, base_var, v, data[v])


def append_nc_var(nc, base_var, metric_var, data):
    if metric_var not in nc.variables:
        dims = nc.variables[base_var].dimensions[1:]
        outVar = nc.createVariable(
            metric_var, nc.variables[base_var].datatype, dims, zlib=True
        )
        outVar.setncatts({"units": nc.variables[base_var].units})
    else:
        outVar = nc.variables[metric_var]
    outVar[...] = data


def get_mask(mask_fn, mask_name):
    """Some options: coast_99, coast_54, coast_27, coast_18, coast_9, shelf_mask_100, shelf_mask_150,
    shelf_mask_200"""
    if mask.endswith('.h5') or mask.endswith('.mat'):
        with h5py.File(mask_fn, "r") as fp:
            amask = fp["analysis_masks"][mask_name][...]
        return np.transpose(amask).astype(bool)
    elif mask.endswith('.nc'):
        ds=xr.open_dataset(mask_fn)
        amask = ds[mask_name][...].values
        ds.close()
        return amask

def build_grid_vars(mask_h5, data_mask, area):

    area = np.loadtxt(r"/mnt/reservior/data/CWM/grid/area_twelfth_degree.txt")
    ds = create_cwm_annual_dataset([area], ["m^2"])
    # ds[geoname][...] = eez1
    ds["MRGID"] = feature["properties"]["MRGID"]
    ds["SOVEREIGN1"] = feature["properties"]["SOVEREIGN1"]
    ds["TERRITORY1"] = feature["properties"]["TERRITORY1"]
    encoding = {geoname: {"dtype": "i1", "zlib": True, "complevel": 4}}
    ds.to_netcdf(out_nc_path, format="netCDF4", encoding=encoding)

    for mask_name in [
        "coast_99",
        "coast_54",
        "coast_27",
        "coast_18",
        "coast_9",
        "shelf_mask_100",
        "shelf_mask_150",
        "shelf_mask_200",
    ]:
        mask_data = get_mask(mask_h5, mask_name)
        # load into netcdf
        open_append_nc(output_file, base_var, metric_var_list, data)


# xesmf cannot handle large data.  lame
def generate_ship_traffic_lame(input_geotiff, method="bilinear"):

    from osgeo import gdal, gdalconst
    import xesmf as xe

    # Source
    src = gdal.Open(input_geotiff, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()

    # gt the coordinates of the corners:
    width = src.RasterXSize
    height = src.RasterYSize
    gt = src.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    # make regridder imput xarray
    var_dict = {
        "ship_density": xr.DataArray(
            dims=["latitude", "longitude"],
            coords={
                "latitude": np.arange(33998) * 0.005 - 84.9873520629999177 + 0.0025,
                "longitude": np.arange(72006) * 0.005 - 180.0153112749999877 + 0.0025,
            },
            attrs={"_FillValue": 2.14748e09, "units": "#AIS"},
            # data=np.array(src.GetRasterBand(1).ReadAsArray())
        )
    }
    ds_in = xr.Dataset(var_dict)

    latb = np.arange(33999) * 0.005 - 84.9873520629999177
    lonb = np.arange(72007) * 0.005 - 180.0153112749999877

    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()
    encoding = {"ship_density_cwm": {"dtype": np.float32, "zlib": True, "complevel": 4}}
    var_dict = {
        "ship_density_cwm": xr.DataArray(
            dims=["latitude", "longitude"],
            coords={"latitude": latitude, "longitude": longitude},
            attrs={"_FillValue": -999.9, "units": "km"},
        )
    }
    ds_out = xr.Dataset(var_dict)

    regridder = xe.Regridder(ds_in, ds_out, method)
    sd_interp = regridder(src.GetRasterBand(1).ReadAsArray())
    ds_out.ship_density_cwm[...] = sd_interp

    ds_out.to_netcdf(
        r"X:\data\CWM\regions_and_masks\Global Ship Density\global_ship_density.nc",
        format="netCDF4",
        encoding=encoding,
    )
    ds_out.close()


def generate_ship_traffic(input_geotiff):

    from osgeo import gdal, gdalconst

    # Source
    src = gdal.Open(input_geotiff, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()

    # gt the coordinates of the corners:
    width = src.RasterXSize
    height = src.RasterYSize
    gt = src.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    nlon = 72006
    nlat = 33998
    gsize = 0.005000000000000000104

    lat_in = np.arange(nlat) * gsize - 84.9873520629999177 + gsize / 2
    lon_in = np.arange(nlon) * gsize - 180.0153112749999877 + gsize / 2
    # xlonin, ylatin = np.meshgrid(lon_in, lat_in)
    (
        longitude,
        latitude,
        xlon,
        ylat,
        mask,
        area,
        lonb,
        latb,
        xlonb,
        ylatb,
    ) = load_cwm_grid_area_bounds()
    # weights = src.GetRasterBand(1).ReadAsArray()  # 19*4Gb memory

    sd_sum = np.zeros((len(latitude), len(longitude)), np.float32)

    chunks = 5
    xb = int(72006 / chunks)
    yb = int(33998 / chunks)
    for xi in range(chunks):
        x1 = xb * xi
        spanx = xb
        if xi == chunks - 1:
            spanx = nlon - x1
        for yi in range(chunks):
            y1 = yb * yi
            spany = yb
            if yi == chunks - 1:
                spany = nlat - y1

            weights = src.GetRasterBand(1).ReadAsArray(x1, y1, spanx, spany)
            xlonin, ylatin = np.meshgrid(
                lon_in[x1 : x1 + spanx], lat_in[y1 : y1 + spany]
            )

            print("interpolating:", xi, yi, x1, ":", x1 + spanx, y1, ":", y1 + spany)
            sd_interp, _, _ = np.histogram2d(
                ylatin.ravel(),
                xlonin.ravel(),
                bins=(np.flip(latb), lonb),
                weights=weights.ravel(),
            )

            sd_sum += np.flip(sd_interp, axis=0)

    encoding = {"ship_density_cwm": {"dtype": np.float32, "zlib": True, "complevel": 4}}
    var_dict = {
        "ship_density_cwm": xr.DataArray(
            dims=["latitude", "longitude"],
            coords={"latitude": latitude, "longitude": longitude},
            attrs={"_FillValue": -999.9, "units": "km"},
            data=np.flip(sd_sum, axis=0),
        )
    }
    ds_out = xr.Dataset(var_dict)
    ds_out.to_netcdf(
        r"X:\data\CWM\regions_and_masks\Global Ship Density\global_ship_density.nc",
        format="netCDF4",
        encoding=encoding,
        mode="w",
    )
    ds_out.close()


def EEZ_masks(eez_shapefile, out_nc_path):
    import fiona

    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()

    total_eez = np.full(np.shape(xlon), False, dtype=bool)

    fc = fiona.open(eez_shapefile)
    nc = None
    for feature in fc:
        geom = feature["geometry"]
        if geom["type"] == "Polygon" or geom["type"] == "MultiPolygon":
            if geom["type"] == "MultiPolygon":
                coord_set = sum(geom["coordinates"], [])  # flatten
            else:
                coord_set = geom["coordinates"]

            geoname = feature["properties"]["GEONAME"]
            geoname = geoname.replace(r"/", "-")
            geoname = geoname.replace(":", "-")
            geoname = geoname.replace("(", "--")
            geoname = geoname.replace(")", "--")
            print("Processing:", geoname)

            # write as single mask to all-mask netcdf
            not_found = False
            if not os.path.exists(out_nc_path):
                ds = create_cwm_annual_dataset([geoname], ["mask"])
                # ds[geoname][...] = eez1
                ds["MRGID"] = feature["properties"]["MRGID"]
                ds["SOVEREIGN1"] = feature["properties"]["SOVEREIGN1"]
                ds["TERRITORY1"] = feature["properties"]["TERRITORY1"]
                encoding = {geoname: {"dtype": "i1", "zlib": True, "complevel": 4}}
                ds.to_netcdf(out_nc_path, format="netCDF4", encoding=encoding)
                not_found = True

            if nc is None:
                nc = netCDF4.Dataset(out_nc_path, "a")
            if geoname not in nc.variables:
                outVar = nc.createVariable(
                    geoname, "i1", ["latitude", "longitude"], zlib=True
                )
                outVar.setncatts(
                    {
                        "units": "mask",
                        "MRGID": feature["properties"]["MRGID"],
                        "SOVEREIGN1": feature["properties"]["SOVEREIGN1"],
                        "TERRITORY1": feature["properties"]["TERRITORY1"],
                    }
                )
                not_found = True

            outVar = nc.variables[geoname]

            if geoname == "United States Exclusive Economic Zone":
                not_found = True

            if not_found:
                print("Not Found!")
                eez1 = np.full(np.shape(xlon), False, dtype=bool)
                for poly in coord_set:
                    coords = np.asarray(poly)
                    eez1 = np.logical_or(
                        eez1, is_inside_sm_parallel(xlon, ylat, coords)
                    )
                outVar[...] = eez1
                nc.sync()
            else:
                eez1 = outVar[...]

            total_eez = np.logical_or(total_eez, eez1)

    if "eez" not in nc.variables:
        eezvar = nc.createVariable("eez", "i1", ["latitude", "longitude"], zlib=True)
        eezvar.setncatts({"units": "mask"})
    else:
        eezvar = outVar["eez"]
    eezvar[...] = total_eez
    nc.close()
    fc.close()


def EEZ_mask_USA(eez_shapefile, out_nc_path):
    import fiona

    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()

    total_eez = np.full(np.shape(xlon), False, dtype=bool)

    fc = fiona.open(eez_shapefile)
    nc = None
    for feature in fc:
        geom = feature["geometry"]
        if geom["type"] == "Polygon" or geom["type"] == "MultiPolygon":
            if geom["type"] == "MultiPolygon":
                coord_set = sum(geom["coordinates"], [])  # flatten
            else:
                coord_set = geom["coordinates"]

            geoname = feature["properties"]["GEONAME"]
            geoname = geoname.replace(r"/", "-")
            geoname = geoname.replace(":", "-")
            geoname = geoname.replace("(", "--")
            geoname = geoname.replace(")", "--")
            print("Processing:", geoname)

            if nc is None:
                nc = netCDF4.Dataset(out_nc_path, "a")

            eez1 = np.full(np.shape(xlon), False, dtype=bool)
            for poly in coord_set:
                coords = np.asarray(poly)
                eez1 = np.logical_or(
                    eez1, is_inside_sm_parallel(xlon, ylat, coords)
                )
            total_eez = np.logical_or(total_eez, eez1)

    if "eez_usa_all" not in nc.variables:
        eezvar = nc.createVariable("eez_usa_all", "i1", ["latitude", "longitude"], zlib=True)
        eezvar.setncatts({"units": "mask"})
        eezvar.setncatts({"long name": "USA and territories EEZ"})

    eezvar[...] = total_eez
    nc.close()
    fc.close()



def LME66_masks(LME_shapefile, out_nc_path):
    import fiona

    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()

    total_eez = np.full(np.shape(xlon), False, dtype=bool)

    fc = fiona.open(LME_shapefile)
    nc = None
    for feature in fc:
        geom = feature["geometry"]
        if geom["type"] == "Polygon" or geom["type"] == "MultiPolygon":
            if geom["type"] == "MultiPolygon":
                coord_set = sum(geom["coordinates"], [])  # flatten
            else:
                coord_set = geom["coordinates"]

            geoname = feature["properties"]["LME_NAME"]
            geoname = geoname.replace(r"/", "-")
            geoname = geoname.replace(":", "-")
            geoname = geoname.replace("(", "--")
            geoname = geoname.replace(")", "--")
            print("Processing:", geoname)

            # write as single mask to all-mask netcdf
            not_found = False
            if not os.path.exists(out_nc_path):
                ds = create_cwm_annual_dataset([geoname], ["mask"])
                # ds[geoname][...] = eez1
                ds["Shape_Area"] = feature["properties"]["Shape_Area"]
                ds["SUM_GIS_KM"] = feature["properties"]["SUM_GIS_KM"]
                ds["LME_NUMBER"] = feature["properties"]["LME_NUMBER"]
                encoding = {geoname: {"dtype": "i1", "zlib": True, "complevel": 4}}
                ds.to_netcdf(out_nc_path, format="netCDF4", encoding=encoding)
                not_found = True

            if nc is None:
                nc = netCDF4.Dataset(out_nc_path, "a")
            if geoname not in nc.variables:
                outVar = nc.createVariable(
                    geoname, "i1", ["latitude", "longitude"], zlib=True
                )
                outVar.setncatts(
                    {
                        "units": "mask",
                        "Shape_Area": feature["properties"]["Shape_Area"],
                        "SUM_GIS_KM": feature["properties"]["SUM_GIS_KM"],
                        "LME_NUMBER": feature["properties"]["LME_NUMBER"],
                    }
                )
                not_found = True

            outVar = nc.variables[geoname]

            if not_found:
                print("Not Found!")
                eez1 = np.full(np.shape(xlon), False, dtype=bool)
                for poly in coord_set:
                    coords = np.asarray(poly)
                    eez1 = np.logical_or(
                        eez1, is_inside_sm_parallel(xlon, ylat, coords)
                    )
                outVar[...] = eez1
                nc.sync()
            else:
                eez1 = outVar[...]

            total_eez = np.logical_or(total_eez, eez1)

    if "LME_all" not in nc.variables:
        eezvar = nc.createVariable(
            "LME_all", "i1", ["latitude", "longitude"], zlib=True
        )
        eezvar.setncatts({"units": "mask"})
    else:
        eezvar = outVar["LME_all"]
    eezvar[...] = total_eez
    nc.close()
    fc.close()


def MEOW_masks(MEOW_shapefile, out_nc_path):
    import fiona

    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()

    total_eez = np.full(np.shape(xlon), False, dtype=bool)

    fc = fiona.open(MEOW_shapefile)
    nc = None
    for feature in fc:
        geom = feature["geometry"]
        if geom["type"] == "Polygon" or geom["type"] == "MultiPolygon":
            if geom["type"] == "MultiPolygon":
                coord_set = sum(geom["coordinates"], [])  # flatten
            else:
                coord_set = geom["coordinates"]

            geoname = feature["properties"]["ECOREGION"]
            geoname = geoname.replace(r"/", "-")
            geoname = geoname.replace(":", "-")
            geoname = geoname.replace("(", "--")
            geoname = geoname.replace(")", "--")
            print("Processing:", geoname)

            # write as single mask to all-mask netcdf
            not_found = False
            if not os.path.exists(out_nc_path):
                ds = create_cwm_annual_dataset([geoname], ["mask"])
                # ds[geoname][...] = eez1
                ds["PROVINCE"] = feature["properties"]["PROVINCE"]
                ds["REALM"] = feature["properties"]["REALM"]
                ds["Lat_Zone"] = feature["properties"]["Lat_Zone"]
                encoding = {geoname: {"dtype": "i1", "zlib": True, "complevel": 4}}
                ds.to_netcdf(out_nc_path, format="netCDF4", encoding=encoding)
                not_found = True

            if nc is None:
                nc = netCDF4.Dataset(out_nc_path, "a")
            if geoname not in nc.variables:
                outVar = nc.createVariable(
                    geoname, "i1", ["latitude", "longitude"], zlib=True
                )
                outVar.setncatts(
                    {
                        "units": "mask",
                        "PROVINCE": feature["properties"]["PROVINCE"],
                        "REALM": feature["properties"]["REALM"],
                        "Lat_Zone": feature["properties"]["Lat_Zone"],
                    }
                )
                not_found = True

            outVar = nc.variables[geoname]

            if not_found:
                print("Not Found!")
                eez1 = np.full(np.shape(xlon), False, dtype=bool)
                for poly in coord_set:
                    coords = np.asarray(poly)
                    eez1 = np.logical_or(
                        eez1, is_inside_sm_parallel(xlon, ylat, coords)
                    )
                outVar[...] = eez1
                nc.sync()
            else:
                eez1 = outVar[...]

            total_eez = np.logical_or(total_eez, eez1)

    if "MEOW_all" not in nc.variables:
        eezvar = nc.createVariable(
            "MEOW_all", "i1", ["latitude", "longitude"], zlib=True
        )
        eezvar.setncatts({"units": "mask"})
    else:
        eezvar = outVar["MEOW_all"]
    eezvar[...] = total_eez
    nc.close()
    fc.close()


def FAO_masks(FAO_shapefile, out_nc_path):
    import fiona

    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()

    total_eez = np.full(np.shape(xlon), False, dtype=bool)

    fc = fiona.open(FAO_shapefile)
    nc = None
    for feature in fc:
        geom = feature["geometry"]
        if geom["type"] == "Polygon" or geom["type"] == "MultiPolygon":
            if geom["type"] == "MultiPolygon":
                coord_set = sum(geom["coordinates"], [])  # flatten
            else:
                coord_set = geom["coordinates"]
            geoname = feature["properties"]["NAME_EN"]
            if geoname is None:
                geoname = feature["properties"]["F_CODE"]
            geoname = geoname.replace(r", ", " - ")
            geoname = geoname.replace(r"/", "-")
            geoname = geoname.replace(":", "-")
            geoname = geoname.replace("(", "--")
            geoname = geoname.replace(")", "--")
            geoname = feature["properties"]["F_LEVEL"] + " " + geoname
            print("Processing:", geoname)

            # write as single mask to all-mask netcdf
            not_found = False
            if not os.path.exists(out_nc_path):
                ds = create_cwm_annual_dataset([geoname], ["mask"])
                # ds[geoname][...] = eez1
                ds["F_CODE"] = feature["properties"]["F_CODE"]
                ds["F_LEVEL"] = feature["properties"]["F_LEVEL"]
                ds["SUBOCEAN"] = feature["properties"]["SUBOCEAN"]
                ds["F_AREA"] = feature["properties"]["F_AREA"]
                ds["OCEAN"] = feature["properties"]["OCEAN"]
                ds["ID"] = feature["properties"]["ID"]
                encoding = {geoname: {"dtype": "i1", "zlib": True, "complevel": 4}}
                ds.to_netcdf(out_nc_path, format="netCDF4", encoding=encoding)
                not_found = True

            if nc is None:
                nc = netCDF4.Dataset(out_nc_path, "a")
            if geoname not in nc.variables:
                outVar = nc.createVariable(
                    geoname, "i1", ["latitude", "longitude"], zlib=True
                )
                outVar.setncatts(
                    {
                        "units": "mask",
                        "F_CODE": feature["properties"]["F_CODE"],
                        "F_LEVEL": feature["properties"]["F_LEVEL"],
                        "SUBOCEAN": feature["properties"]["SUBOCEAN"],
                        "F_AREA": feature["properties"]["F_AREA"],
                        "OCEAN": feature["properties"]["OCEAN"],
                        "ID": feature["properties"]["ID"],
                    }
                )
                not_found = True

            outVar = nc.variables[geoname]

            if geoname == "United States Exclusive Economic Zone":
                not_found = True

            if not_found:
                print("Not Found!")
                eez1 = np.full(np.shape(xlon), False, dtype=bool)
                for poly in coord_set:
                    coords = np.asarray(poly)
                    eez1 = np.logical_or(
                        eez1, is_inside_sm_parallel(xlon, ylat, coords)
                    )
                outVar[...] = eez1
                nc.sync()
            else:
                eez1 = outVar[...]

            total_eez = np.logical_or(total_eez, eez1)

    if "eez" not in nc.variables:
        eezvar = nc.createVariable("eez", "i1", ["latitude", "longitude"], zlib=True)
        eezvar.setncatts({"units": "mask"})
    else:
        eezvar = outVar["eez"]
    eezvar[...] = total_eez
    nc.close()
    fc.close()


def MPA_mask(shapefile_list, out_nc_path, minimum_sq_km=20.0):
    import fiona

    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()

    mpa_all = np.full(np.shape(xlon), False, dtype=np.bool_)

    for shpf in shapefile_list:
        fc = fiona.open(shpf)
        nfeat = len(fc)
        shpp,fname = os.path.split(shpf)        

        for i,feature in enumerate(fc):
    
            if feature["properties"]["REP_M_AREA"] > minimum_sq_km or \
                feature["properties"]["GIS_M_AREA"] > minimum_sq_km:
    
                geom = feature["geometry"]
                if geom["type"] == "Polygon" or geom["type"] == "MultiPolygon":
                    geoname = feature["properties"]["NAME"]
                    print('Processing %s -- %s, %i of %i'%(fname,geoname,i,nfeat))
                    if geom["type"] == "MultiPolygon":
                        coord_set = sum(geom["coordinates"], [])  # flatten
                    else:
                        coord_set = geom["coordinates"]    
    
                    for poly in coord_set:
                        coords = np.asarray(poly)
                        mpa_all = np.logical_or(
                            mpa_all, is_inside_sm_parallel(xlon, ylat, coords)
                        )
        fc.close()
    
    create_write_cwm_annual_dataset(['mpa_mask_min_20_sq_km'], ["mask"], [mpa_all], out_nc_path, 
                                        dtype_list=["i1"])


@jit(nopython=True)
def is_inside_sm(polygon, point):
    """
    https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
    https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    """
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (
            point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]
        ):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]

                if (
                    point[0] > F
                ):  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (
                point[0] == polygon[jj][0]
                or (
                    dy == 0
                    and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0
                )
            ):
                return 2

        ii = jj
        jj += 1

    # print 'intersections =', intersections
    return intersections & 1


@njit(parallel=True)
def is_inside_sm_parallel(grid_x_lon, grid_y_lat, polygon):
    nx, ny = np.shape(grid_x_lon)
    D = np.full(np.shape(grid_x_lon), False, dtype=np.bool_)
    for i in numba.prange(nx):
        for j in range(ny):
            D[i, j] = is_inside_sm(polygon, (grid_x_lon[i, j], grid_y_lat[i, j]))
    return D


class Nutrient8Day(object):
    """Find potential nutrient concentrations by:
    1) Calculate nutrients from phytoplankton stoichiometry and satellite-based
    production estimates.
    2) Constrain nutrient flux to the mixed layer with a combination of model-
    derived upwelling rates, and some sort of time constant.
    3) Constrain by gridded WOCE nutrient concentrations, in some fashion.
    4) Constrain  by maximum phytoplankton growth rates (but production is
     already constained by that?) somehow?

    """

    def __init__(self, production, C2N, C2P=None, spread_type="latutude"):
        pass

    def latitude_spread(self, lat_0_spread=14, lat_80_spread=30 * 8):
        """Function the spreads production-derived nutrients backwards, based on latitude,
        where tropical nutrients are assumed to be upwelled and consumed quickly, while
        at high-latitudes production-based nutrients are assumed to tyo have been
        upwelled over a longer period (b/c/. no winter production)"""

    def constant_spread(self, spread_days=21):
        """Function the spreads production-derived nutrients backwards by 21 days, which can
        be though of as the maximum time it might take for a phytoplankton bloom to develop.
        This could be though of as part of a more conservative nutrient calc.  The risk is
        that nturients might not be calculauted as avaiable in temperature late-winter (like
        Feb in N.H.) when seaweeds get a jump-start on growth compared to phytoplankton.
        """

    def production_spread(self, spread_days=21):
        """Function the spreads production-derived nutrients backwards to the end of
        the previous bloom in the *region* (what is a region?).
        """


def get_grid_vertices(sixth_degree=False):
    m = Basemap(
        projection="cyl",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=-180,
        urcrnrlon=180,
        resolution="c",
    )
    if sixth_degree:
        return np.asarray(m.makegrid(2161, 1081))
    else:
        return np.asarray(m.makegrid(4321, 2161))  # vertex coordinates


def nansum_old(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    a, mask = _replace_nan(a, 0)
    if mask is None:
        return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    mask = np.all(mask, axis=axis, keepdims=keepdims)
    tot = np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    if np.any(mask):
        tot = _copyto(tot, np.nan, mask)
        # warnings.warn("In Numpy 1.9 the sum along empty slices will be zero.",
        #              FutureWarning)
    return tot


def nanmean_old(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    a, mask = _replace_nan(a, 0)
    if mask is None:
        return np.mean(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    mask = np.all(mask, axis=axis, keepdims=keepdims)
    tot = np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    if np.any(mask):
        tot = _copyto(tot, np.nan, mask)
        # warnings.warn("In Numpy 1.9 the sum along empty slices will be zero.",
        #              FutureWarning)
    return tot


def make_mask_from_hycom(hycom_filepath,save_fname):
    """They already figured with out at a higher resolution that 9km ... idea is to get nearest neigbor from a 2D
    interpolation, preserving NaN if possible"""

    # read input coords and array sizes
    longitude, latitude, xlon, ylat, mask = load_cwm_grid()
    #create_write_cwm_annual_dataset(["mask"], ["0to1"], [mask], save_fname + '.nc', dtype_list=None)
    #exit()
    lon_mod = longitude
    lon_mod[
        -1
    ] = 179.92  # regular grid interp can't handle last value, somehow outside bounds
    xlon, ylat = np.meshgrid(
        lon_mod, latitude[0:2039]
    )  # constrain to input grid which ends at -80 latitude
    ds_in = xr.load_dataset(hycom_filepath)
    lon_in = ds_in.lon[:].values
    positive_lon = False
    if np.max(lon_in) > 190:
        # grid is 0:365 instead of -180:180, we need -180:180 so will need to re-arange array below
        lon_in = lon_in - 180.0
        positive_lon = True
    lat_in = ds_in.lat[:].values  # input coordinates

    water_u = ds_in.water_u[0,0,:].values
    water_u = water_u.astype(np.float32)
    water_u[water_u==-30000] = np.nan

    dude = RegularGridInterpolator(
        (lat_in, lon_in),
        center_on_meridian_hycom_or_not(water_u, positive_lon),
    )
    mask = np.full((2160,4320),np.nan)
    mask[:2039,:] = dude((ylat, xlon), 'nearest')
    mask[np.isfinite(mask)] = 1;
    mask[np.isnan(mask)] = 0;
    #mask = mask.astype(np.int)
    if save_fname is not None:
        np.savetxt(save_fname, mask.astype(np.int), fmt="%1i")
        fn,_ = os.path.splitext(save_fname)
        create_write_cwm_annual_dataset(["mask"], ["0to1"], [mask], fn+'.nc', dtype_list=None)
        hf = h5py.File(fn+'.h5', 'w')
        hf.create_dataset('cwm_ocean_mask', data=mask.astype(np.int),compression="gzip")
        hf.close()
    return mask




def make_mask(directory, type="cbpm", years=list(range(2003, 2020)), save_fname=None):
    """Look over all values in 8-day files to determine what is a valid pixel. type could also reasonably be
    'sst' since where there is water, this sould work. Need to mask out great lakes and some other areas
    maybe some other way.  Use many years avoid spots with big icebergs, etc."""
    mask = np.full((2160, 4320), np.nan)
    for year in years:
        for i in range(1, 362, 8):
            fn = os.path.join(directory, "%s.%i%03i.hdf" % (type, year, i))
            print("reading %s" % fn)
            file = SD(fn, SDC.READ)
            data = file.select("npp").get()  # select sds and get sds data
            data[data == -9999.0] = np.nan  # nan-ify missing values
            # mask = np.nansum(np.dstack((mask,data)),2) # crap nansum behavior changed to return zeros!!!
            mask = nansum_old(np.dstack((mask, data)), 2)
    mask[np.isfinite(mask)] = 1
    mask[np.isnan(mask)] = 0
    mask = mask.astype(np.int)
    if save_fname is not None:
        np.savetxt(save_fname, mask, fmt="%1i")
    return mask


def Galbraith_Martiny_C_to_N(NO3):
    """Galbraith, Eric D., and Adam C. Martiny. 2015. “A simple nutrient-dependence mechanism for predicting the
    stoichiometry of marine ecosystems.” PNAS 112 (27): 8199–8204. https://doi.org/10.1073/pnas.1423917112.
    N:C = 125 ‰ + 30 ‰ × NO3 / (0.32 μM + NO3)
    """
    return 1.0 / (0.125 + 0.03 * NO3 / (0.32 + NO3))  # with NO3 in μM


def get_behrenfeld_hdf(fname, varname, print_report=True, use_nc=False):
    if print_report:
        print("reading %s" % fname)
    if use_nc:
        path, fn = os.path.split(fname)
        fn_tokens = fn.split(".")
        nc_fname = "%s_%s.nc" % (fn_tokens[0], fn_tokens[-2][0:4])
        ds = xr.open_dataset(nc_fname)
        day = int(fn_tokens[-2][4:])
        data = ds[fn_tokens[0]][day, ...]
    else:
        file = SD(fname, SDC.READ)
        data = file.select(varname).get()  # select sds and get sds data
        data[data == -9999.0] = np.nan  # nan-ify missing values
    return data


def expand_tarfile(tf, path_for_uncompress):
    import tarfile

    my_tar = tarfile.open(tf)
    my_tar.extractall(path_for_uncompress)  # specify which folder to extract to
    my_tar.close()


def ungzip(gz_file, new_fn=None, del_gz=True):
    import gzip
    import shutil

    if new_fn is not None:
        outfile = new_fn
    else:
        outfile, ext = os.path.splitext(gz_file)
        if ext != ".gz":
            raise ValueError("Not sure what to name un-gzipped file")
    with gzip.open(gz_file, "rb") as f_in:
        with open(outfile, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    if del_gz:
        os.remove(gz_file)


def expand_unzip_bh_tarfile(var_path, varname, year):
    if varname == "mld":
        tf = os.path.join(var_path, "%s.hycom_125.%i.tar" % (varname, year))
    else:
        tf = os.path.join(var_path, "%s.m.%i.tar" % (varname, year))
    print("Trying to extract TAR...")
    expand_tarfile(tf, var_path)
    for i, day in enumerate(range(1, 362, 8)):
        gz_fname = os.path.join(var_path, "%s.%i%03i.hdf.gz" % (varname, year, day))
        ungzip(gz_fname)


def generate_behrenfeld_nc(year, varname, units):
    bh_dirs = {
        "cbpm": "production",
        "mld": "MLD",
        "par": "PAR",
        "sst": "SST",
        "chl": "chla",
    }
    var_path = os.path.join(r"X:\data\CWM", bh_dirs[varname])
    nc_fname = os.path.join(var_path, "%s_%i.nc" % (varname, year))

    hdf_varname = "npp" if varname == "cbpm" else varname

    # create dataset
    bh_var = [varname]
    ds = create_cwm_output_dataset(year, bh_var, [units])
    encoding = {
        v: {
            "dtype": np.float32,
            "zlib": True,
            "complevel": 4,
            "chunksizes": (1, 360, 4320),
        }
        for v in bh_var
    }
    ds.to_netcdf(nc_fname, format="netCDF4", encoding=encoding)
    longitude, latitude, xlon, ylat, mask = load_cwm_grid()

    rg = netCDF4.Dataset(nc_fname, "a")
    for i, day in enumerate(range(1, 362, 8)):
        hdf_fname = os.path.join(var_path, "%s.%i%03i.hdf" % (varname, year, day))
        if not os.path.exists(hdf_fname):
            try:
                expand_unzip_bh_tarfile(var_path, varname, year)
            except:
                print("Failed to extract tar.")
        rg.variables[varname][i, ...] = get_behrenfeld_hdf(hdf_fname, hdf_varname)
        rg.sync()
    rg.close()


def nitrogen_static(base_dir, year, C_to_N=16.0 / 106.0):

    # setup time dimension
    dtt = [
        datetime.datetime(year - 1, 12, 31) + datetime.timedelta(days=i)
        for i in range(1, 362, 8)
    ]
    time = pd.to_datetime(dtt)

    # setup horizontal dimensions
    longitude = np.arange(4320) * 1 / 12 - 180 + 1 / 24
    latitude = -1.0 * (np.arange(2160) * 1 / 12 - 90 + 1 / 24)

    # create dataset
    ds = xr.Dataset(
        {
            "N": xr.DataArray(
                # data=np.random.random(6),  # enter data here
                dims=["time", "latitude", "longitude"],
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                attrs={"_FillValue": -999.9, "units": "mg N m-3 day-1"},
            )
        }
    )

    # calc and fill data
    production_dir = os.path.join(base_dir, "production")
    mld_dir = os.path.join(base_dir, "MLD")
    for i, day in enumerate(range(1, 362, 8)):
        npp = get_behrenfeld_hdf(
            os.path.join(production_dir, "cbpm.%i%03i.hdf" % (year, day)), "npp"
        )
        npp[
            npp > 5000
        ] = 5000  # some values are way to high.  What is a valid maximum? 5g/day is high, maybe 10x that?
        mld = get_behrenfeld_hdf(
            os.path.join(mld_dir, "mld.%i%03i.hdf" % (year, day)), "mld"
        )
        N_m2_d = npp * C_to_N  # mgC m-2 d-1 -> mg N m-2 day-1
        ds.N[i, :, :] = N_m2_d / mld  # mgN m-2 d-1 -> mg N m-3 day-1

    # save
    out_dir = os.path.join(base_dir, "nitrogen")
    outfile = os.path.join(out_dir, "N_static_%i.nc" % year)
    ds.to_netcdf(
        outfile,
        format="netCDF4",
        encoding={
            "N": {
                "dtype": np.float32,
                "zlib": True,
                "complevel": 4,
                "chunksizes": (1, 360, 4320),
            }
        },
    )


def map_9km_to_WOA():
    lon = np.loadtxt("/Volumes/Aux/more_data/CWM/grid/centers_lon_sixth_deg.txt")
    lat = np.loadtxt("/Volumes/Aux/more_data/CWM/grid/centers_lon_sixth_deg.txt")
    i = np.round(lon) + 180
    j = np.round(lat) + 90
    np.savetxt("/Volumes/Aux/more_data/CWM/grid/9km_to_WOA_i.txt", i, fmt="%3i")
    np.savetxt("/Volumes/Aux/more_data/CWM/grid/9km_to_WOA_j.txt", j, fmt="%3i")

@jit(nopython=True)
def tracer_weighted_mean(tracer,layer_th):
    nz, ny, nx = np.shape(tracer)
    tracer_mean = np.full((ny,nx),np.nan,np.float32)
    for i in range(ny):
        for j in range(nx):
            mask = np.isfinite(tracer[:,i,j])
            if np.any(mask):
                m1 = tracer[:,i,j]
                m1 = m1[mask]
                t1 = layer_th[mask]
                tracer_mean[i,j] = np.sum(m1*t1)/np.sum(t1)
    return tracer_mean

@jit('f4[:,:](f4[:,:,:], f8[:], f8)',nopython=True)
def tracer_weighted_mean_partial_layers(tracer,layer_th,total_th):
    nz, ny, nx = np.shape(tracer)
    tracer_mean = np.full((ny,nx),np.nan,np.float32)
    for i in range(ny):
        for j in range(nx):
            zth = 0.0
            zsum = 0.0
            tzsum = 0.0
            mask = np.isfinite(tracer[:, i, j])
            if np.any(mask):
                for k in range(nz):
                    znext = zth + layer_th[k]
                    if znext > total_th:
                        # done, target depth reached
                        if np.isfinite(tracer[k,i,j]):
                            dz = (znext - total_th)
                            tzsum += dz*tracer[k,i,j]
                            zsum += dz
                        if zsum > 0.:
                            tracer_mean[i,j] = tzsum/zsum
                            break
                    else:
                        if np.isfinite(tracer[k,i,j]):
                            tzsum += dz*tracer[k,i,j]
                            zsum += dz
                        zth = znext
    return tracer_mean



def load_WOA_annual(woa_path, depth_mean=20, var='n'):

    WOA = np.full((12, 180, 360), np.nan)
    for i in range(12):
        # xarray choking on non-compliant or weird time variable. sigh.
        # xg = xr.open_dataset(os.path.join(woa_path,'woa13_all_n%02i_01.nc'%i))
        # WOA_N[...,i] = xg['n_an'][:]
        rg = netCDF4.Dataset(os.path.join(woa_path, "woa18_all_%s%02i_01.nc" % (var,i + 1)))
        db = rg.variables['depth_bnds'][...]
        zth = np.array([db[i][1]-db[i][0] for i in range(len(db))],dtype=np.float64)
        woa_data = rg.variables["%s_an"%var][0, ...].filled(np.nan)
        WOA[i, ...] = tracer_weighted_mean_partial_layers(woa_data,zth,depth_mean)
    return WOA


def interp_2D_w_nan(arr1, arr2, t1, t2, ti):
    nx, ny = np.shape(arr1)
    result = np.full((nx, ny), np.nan)
    for i in range(nx):
        for j in range(ny):
            if np.isnan(arr1[i, j]):
                result[i, j] = arr1[i, j]
            elif np.isnan(arr2[i, j]):
                result[i, j] = arr2[i, j]
            else:
                m = (arr2[i, j] - arr1[i, j]) / (t2 - t1)
                b = arr1[i, j] - t1 * m
                result[i, j] = ti * m + b
    return result


def interp_2D_w_nan_scipy(arr_3D, t, ti, t1i, t2i, kind='cubic'):
    nt, nx, ny = np.shape(arr_3D)
    result = np.full((nx, ny), np.nan)

    # roll the array to center on ti
    shift = t1i - int(len(t) / 2)
    arr_c = np.roll(arr_3D, shift, axis=0)
    t_c = np.roll(t,shift)
    if shift != 0:
        t_c[shift:] += 365
        if ti < t_c[0]:
            ti += 365

    for i in range(nx):
        #print('interp 2D: row %i of %i'%(i,nx))
        for j in range(ny):
            if np.isnan(arr_3D[t1i,i,j]) or np.isnan(arr_3D[t2i,i,j]):
                result[i, j] = np.nan
            else:
                m = np.isfinite(arr_c[:,i,j])
                interpArr = spint.interp1d(t_c[m], arr_c[:,i,j][m], kind=kind, copy=False, bounds_error=None,
                               assume_sorted=True)
                result[i, j] = interpArr(ti)
    return result

#@jit(nopython=True)
# def interp_2D_w_nan_fast_interp(arr_3D, t, ti, t1i, t2i):
#     '''Run me before this to compile for numba'''
#     nt, nx, ny = np.shape(arr_3D)
#     result = np.full((nx, ny), np.nan)

#     # roll the array to center on ti
#     shift = t1i - int(len(t) / 2)
#     arr_c = np.roll(arr_3D, shift, axis=0)
#     t_c = np.roll(t,shift)
#     if shift != 0:
#         t_c[shift:] += 365
#         if ti < t_c[0]:
#             ti += 365

#     for i in range(nx):
#         #print('interp 2D: row %i of %i'%(i,nx))
#         for j in range(ny):
#             if np.isnan(arr_3D[t1i,i,j]) or np.isnan(arr_3D[t2i,i,j]):
#                 result[i, j] = np.nan
#             else:
#                 m = np.isfinite(arr_c[:,i,j])
#                 interpArr = interp1d(t_c[0],t_c[-1],t_c[m],arr_c[:,i,j][m], k=3)
#                 result[i, j] = interpArr(ti)
#     return result

def generate_WOA_8day(woa_path, write_out=True, convert_from_per_kg=True,cubic=False,fast_i=True,depth=20.0,n_or_p='n'):
    WOA_N = load_WOA_annual(woa_path, depth_mean=depth,var=n_or_p)
    outvar = 'P' if n_or_p == 'p' else 'N'
    eight_day_i = np.arange(1, 366, 8)
    monthly_day = [
        15.5,
        31 + 14,
        59 + 15.5,
        90 + 15,
        120 + 15.5,
        151 + 15,
        181 + 15.5,
        212 + 15.5,
        243 + 15,
        273 + 15.5,
        304 + 15,
        334 + 15.5,
    ]
    W8 = np.full((len(eight_day_i), 180, 360), np.nan)
    for i, day in enumerate(eight_day_i):
        # interpolate in time
        tdiff = monthly_day - day
        t1i = np.argmin(np.abs(tdiff))
        if tdiff[t1i] > 0:
            t1i = t1i - 1
        t2i = t1i + 1
        if t1i < 0:  # fix me to use/ load WOA from different year
            t1 = -15.5
            t2 = monthly_day[t1i + 1]
        elif t1i == 11:
            t1 = monthly_day[t1i]
            t2 = 365 + 15.5
            t2i = 0
        else:
            t1 = monthly_day[t1i]
            t2 = monthly_day[t1i + 1]
        print(i, (t1i, t2i))
        if cubic:
            if fast_i:
                W8[i, ...] = interp_2D_w_nan_fast_interp(WOA_N, monthly_day, day, t1i,t2i)
            else:
                W8[i, ...] = interp_2D_w_nan_scipy(WOA_N, monthly_day, day, t1i,t2i, kind='cubic')
        else:
            W8[i, ...] = interp_2D_w_nan(WOA_N[t1i, ...], WOA_N[t2i, ...], t1, t2, day)

    if convert_from_per_kg:
        # see https://www.nodc.noaa.gov/OC5/WOD/wod18-notes.html
        W8 = W8 * 1.0250  # micromol/kg * 1.025kg/l = micromol/l or mmol/m3

    if write_out:
        # setup time dimension
        dtt = [
            datetime.datetime(2003 - 1, 12, 31) + datetime.timedelta(days=i)
            for i in range(1, 362, 8)
        ]
        time = pd.to_datetime(dtt)

        # setup horizontal dimensions
        longitude = np.arange(-180 + 0.5, 180)
        latitude = np.arange(-90 + 0.5, 90)

        # create dataset
        ds = xr.Dataset(
            {
                outvar: xr.DataArray(
                    data=W8,
                    dims=["time", "latitude", "longitude"],
                    coords={
                        "time": time,
                        "latitude": latitude,
                        "longitude": longitude,
                    },
                    attrs={"_FillValue": -999.9, "units": "mmol %s m-3"%outvar},
                )
            }
        )
        if cubic:
            if fast_i:
                fn = "WOA18_%s_8day_interp_fast_cubic_%im.nc"%(outvar,depth)
            else:
                fn = "WOA18_%s_8day_interp_cubic_%im.nc"%(outvar,depth)
        else:
            fn = "WOA18_%s_8day_interp_%im.nc"%(outvar,depth)
        ds.to_netcdf(os.path.join(woa_path, fn))
    return W8


def combine_WOA(woa_path, convert_from_per_kg=True, var='n',depth_mean=20.):
    WOA_N = load_WOA_annual(woa_path, depth_mean=depth_mean,var=var)
    outvar = 'P' if var == 'p' else 'N'

    monthly_day = [
        15.5,
        31 + 14,
        59 + 15.5,
        90 + 15,
        120 + 15.5,
        151 + 15,
        181 + 15.5,
        212 + 15.5,
        243 + 15,
        273 + 15.5,
        304 + 15,
        334 + 15.5,
    ]
    if convert_from_per_kg:
        # see https://www.nodc.noaa.gov/OC5/WOD/wod18-notes.html
        WOA_N = WOA_N * 1.0250  # micromol/kg * 1.025kg/l = micromol/l or mmol/m3

    # setup time dimension
    dtt = [
        datetime.datetime(2018, 1, 1) + datetime.timedelta(days=i-1)
        for i in monthly_day
    ]
    time = pd.to_datetime(dtt)

    # setup horizontal dimensions
    longitude = np.arange(-180 + 0.5, 180)
    latitude = np.arange(-90 + 0.5, 90)

    # create dataset
    ds = xr.Dataset(
        {
            outvar: xr.DataArray(
                data=WOA_N,
                dims=["time", "latitude", "longitude"],
                coords={
                    "time": time,
                    "latitude": latitude,
                    "longitude": longitude,
                },
                attrs={"_FillValue": -999.9, "units": "mmol %s m-3"%outvar},
            )
        }
    )
    fn = "WOA18_%s_%im.nc"%(outvar,depth_mean)
    ds.to_netcdf(os.path.join(woa_path, fn),)



def read_matlab_outfile(outfile, varname, groupname=None):
    f = h5py.File(outfile, "r")
    if groupname is None:
        return f[varname][...]
    else:
        return f[groupname][varname][...]


def load_cwm_grid_no_mask():
    longitude = np.arange(4320) * 1 / 12 - 180 + 1 / 24
    latitude = -1.0 * (np.arange(2160) * 1 / 12 - 90 + 1 / 24)
    xlon, ylat = np.meshgrid(longitude, latitude)  # output coordinate meshes
    return longitude, latitude, xlon, ylat


def load_cwm_grid(mask_fn=default_cwm_mask):
    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()
    nc_mask = mask_fn[:-4]+'.nc'
    if os.path.exists(nc_mask):
        ds = xr.open_dataset(nc_mask)
        mask = np.logical_not(ds['mask'][...].values.astype(np.bool_))
        ds.close()
    else:
        mask = np.logical_not(np.loadtxt(mask_fn))
    return longitude, latitude, xlon, ylat, mask


def load_cwm_grid_area_bounds(mask_fn=default_cwm_mask):
    longitude, latitude, xlon, ylat, mask = load_cwm_grid(mask_fn)
    lonb = np.arange(4321) * 1 / 12 - 180
    latb = -1.0 * (np.arange(2161) * 1 / 12 - 90)
    xlonb, ylatb = np.meshgrid(lonb, latb)
    area = np.loadtxt(default_cwm_grid_area)
    return longitude, latitude, xlon, ylat, mask, area, lonb, latb, xlonb, ylatb


def eight_day_weeks():
    return np.arange(1, 362, 8)


def generate_WOA_9km(woa_8day_fn, out_fn, method="cubic",in_var='N',out_var="NO3",out_units="mg NO3 m-3"):
    # interpolate to 365 (366?) days
    # interpolate each day to 4320x2160 nearest neighbor - maybe just store grid cells of nearest neighbor?
    # then need function to map 4320x2160 to WOA 1 degree w/ mask...
    ds_in = xr.load_dataset(woa_8day_fn)

    # create dataset
    ds = create_cwm_output_dataset(2003, [out_var], [out_units])

    # prep input coordinates input
    longitude, latitude, xlon, ylat, mask = load_cwm_grid()
    xin, yin = np.meshgrid(ds_in.longitude[:], ds_in.latitude[:])  # input coordinates
    xin = xin.flatten()
    yin = yin.flatten()

    for i, day in enumerate(range(1, 362, 8)):
        print("working on week-day:", i, day)
        data = ds_in[in_var][i, :, :].values.flatten()  # array should already have filled with _FillValue with NaN
        valid = np.isfinite(data)
        data_9km = griddata(
            (xin[valid], yin[valid]), data[valid], (xlon, ylat), method=method
        )
        data_9km[mask] = np.nan
        ds[out_var][i, :, :] = data_9km

    ds.to_netcdf(
        out_fn,
        format="netCDF4",
        encoding={
            out_var: {
                "dtype": np.float32,
                "zlib": True,
                "complevel": 4,
                "chunksizes": (1, 360, 4320),
            }
        },
    )
    return ds

def generate_WOA_9km_monthly(woa_mo_fn, out_fn, method="cubic",in_var='N',out_var="NO3",out_units="mg NO3 m-3"):
    # interpolate to 365 (366?) days
    # interpolate each day to 4320x2160 nearest neighbor - maybe just store grid cells of nearest neighbor?
    # then need function to map 4320x2160 to WOA 1 degree w/ mask...
    ds_in = xr.load_dataset(woa_mo_fn)

    # create dataset
    ds = create_cwm_output_dataset(2003, [out_var], [out_units],monthly=True)

    # prep input coordinates input
    longitude, latitude, xlon, ylat, mask = load_cwm_grid()
    xin, yin = np.meshgrid(ds_in.longitude[:], ds_in.latitude[:])  # input coordinates
    xin = xin.flatten()
    yin = yin.flatten()

    for i, day in enumerate(monthly_day_start):
        print("working on mo-day:", i, day)
        data = ds_in[in_var][i,...].values.flatten()  # array should already have filled with _FillValue with NaN
        valid = np.isfinite(data)
        data_9km = griddata(
            (xin[valid], yin[valid]), data[valid], (xlon, ylat), method=method
        )
        data_9km[mask] = np.nan
        ds[out_var][i, ...] = data_9km

    ds.to_netcdf(
        out_fn,
        format="netCDF4",
        encoding={
            out_var: {
                "dtype": np.float32,
                "zlib": True,
                "complevel": 4,
                "chunksizes": (1, 360, 4320),
            }
        },
    )
    return ds


def create_cwm_output_dataset(year,var_list,unit_list,monthly=False):
    # create dataset
    if monthly:
        dtt = [datetime.datetime(year-1,12,31) + datetime.timedelta(days=i) for i in monthly_day_start]
    else:
        dtt = [datetime.datetime(year-1,12,31) + datetime.timedelta(days=i) for i in range(1,362,8)]
    time = pd.to_datetime(dtt)
    longitude = np.arange(4320) * 1/12 - 180 + 1/24
    latitude = -1.0*(np.arange(2160) * 1/12 - 90 + 1/24)
    var_dict = {}
    for var,unit in zip(var_list,unit_list):
        var_dict[var] = xr.DataArray(
            dims=['time','latitude','longitude'],
            coords={'time': time, 'latitude': latitude, 'longitude': longitude},
            attrs={'_FillValue': -999.9,'units': unit}
        )
    return xr.Dataset(var_dict)

def create_cwm_annual_dataset(year,var_list,unit_list,monthly=False):
    '''I guess we have two names for this?'''
    return create_cwm_output_dataset(year,var_list,unit_list,monthly=monthly)


def create_cwm_annual_dataset_netCDF4(var_list, unit_list, nc_file, close=False):
    '''This one will create the dataset without needing to allocate dataarrays
    like when using xarray'''
    root_grp = netCDF4.Dataset(nc_file, 'w', format='NETCDF4')

    # dimensions
    root_grp.createDimension('latitude', 2160)
    root_grp.createDimension('longitude', 4320)

    # variables
    lat = root_grp.createVariable('latitude', 'f4', ('latitude',))
    lon = root_grp.createVariable('longitude', 'f4', ('longitude',))
    for var, unit in zip(var_list, unit_list):
        field = root_grp.createVariable(var, 'f4', ('latitude', 'longitude',), zlib=True,
                                        chunksizes=(1, 360, 4320), fill_value=-999.9)
        field.units = unit
    lon[...] = np.arange(4320) * 1 / 12 - 180 + 1 / 24
    lat[...] = -1.0 * (np.arange(2160) * 1 / 12 - 90 + 1 / 24)
    if close:
        root_grp.close()
        return None
    else:
        return root_grp


def create_write_cwm_annual_dataset(var_list, unit_list, data_list, outfile, 
                                    dtype_list=None, monthly=False):
    ds = create_cwm_annual_dataset(var_list, unit_list, monthly=monthly)
    for v, d in zip(var_list, data_list):
        ds[v][...] = d
    my_dtype_list = [np.float32] * len(var_list) if dtype_list is None else dtype_list
    encoding = {
        v: {"dtype": dtp, "zlib": True, "complevel": 4}
        for v, dtp in zip(var_list, my_dtype_list)
    }
    ds.to_netcdf(outfile, format="netCDF4", encoding=encoding)


def create_cwm_monthly_dataset(year, var_list, unit_list):
    # create dataset
    dtt = [datetime.datetime(year, i, 15) for i in range(1, 13)]
    time = pd.to_datetime(dtt)
    longitude = np.arange(4320) * 1 / 12 - 180 + 1 / 24
    latitude = -1.0 * (np.arange(2160) * 1 / 12 - 90 + 1 / 24)
    var_dict = {}
    for var, unit in zip(var_list, unit_list):
        var_dict[var] = xr.DataArray(
            dims=["time", "latitude", "longitude"],
            coords={"time": time, "latitude": latitude, "longitude": longitude},
            attrs={"_FillValue": -999.9, "units": unit},
        )
    return xr.Dataset(var_dict)




def create_cwm_1day_dataset_netCDF4(year, var_list, unit_list, nc_file, close=False):
    '''This one will create the dataset without needing to allocate dataarrays
    like when using xarray'''
    root_grp = netCDF4.Dataset(nc_file, 'w', format='NETCDF4')
    
    # dimensions
    root_grp.createDimension('time', 365)
    root_grp.createDimension('latitude', 2160)
    root_grp.createDimension('longitude', 4320)
    
    # variables
    time = root_grp.createVariable('time', 'f8', ('time',))
    time.units="days since 2003-01-01 00:00:00"
    time.calendar="proleptic_gregorian"
    lat = root_grp.createVariable('latitude', 'f8', ('latitude',))
    lon = root_grp.createVariable('longitude', 'f8', ('longitude',))
    for var, unit in zip(var_list, unit_list):
        field = root_grp.createVariable(var, 'f8', ('time', 'latitude', 'longitude',),zlib=True,
                                        chunksizes=(1,360,4320),fill_value=-999.9)
        field.units=unit
    time[...] = np.arange(0,365)
    lon[...] = np.arange(4320) * 1 / 12 - 180 + 1 / 24
    lat[...] = -1.0 * (np.arange(2160) * 1 / 12 - 90 + 1 / 24)
    if close:
        root_grp.close()
        return None
    else:
        return root_grp
    


def create_cwm_1day_dataset(year, var_list, unit_list):
    # create dataset
    dtt = [
        datetime.datetime(year - 1, 12, 31) + datetime.timedelta(days=i)
        for i in range(1, 366)
    ]
    time = pd.to_datetime(dtt)
    longitude = np.arange(4320) * 1 / 12 - 180 + 1 / 24
    latitude = -1.0 * (np.arange(2160) * 1 / 12 - 90 + 1 / 24)
    var_dict = {}
    for var, unit in zip(var_list, unit_list):
        var_dict[var] = xr.DataArray(
            dims=["time", "latitude", "longitude"],
            coords={"time": time, "latitude": latitude, "longitude": longitude},
            attrs={"_FillValue": -999.9, "units": unit},
        )
    return xr.Dataset(var_dict)


def create_cwm_5day_dataset(year, var_list, unit_list):
    # create dataset
    dtt = [
        datetime.datetime(year - 1, 12, 31) + datetime.timedelta(days=i)
        for i in range(1, 366, 5)
    ]
    time = pd.to_datetime(dtt)
    dtt = [
        datetime.datetime(year - 1, 12, 31) + datetime.timedelta(days=i)
        for i in range(1, 365, 13)
    ]
    time_13 = pd.to_datetime(dtt)
    dtt = [
        datetime.datetime(year - 1, 12, 31) + datetime.timedelta(days=i)
        for i in range(1, 365, 26)
    ]
    time_26 = pd.to_datetime(dtt)
    dtt = [
        datetime.datetime(year - 1, 12, 31) + datetime.timedelta(days=i)
        for i in range(1, 365, 52)
    ]
    time_52 = pd.to_datetime(dtt)
    longitude = np.arange(4320) * 1 / 12 - 180 + 1 / 24
    latitude = -1.0 * (np.arange(2160) * 1 / 12 - 90 + 1 / 24)
    var_dict = {}
    for var, unit in zip(var_list, unit_list):
        if "_13_day" in var:
            tname = "time_13"
            t = time_13
        elif "_26_day" in var:
            tname = "time_26"
            t = time_26
        elif "_52_day" in var:
            tname = "time_52"
            t = time_52
        else:
            tname = "time"
            t = time
        var_dict[var] = xr.DataArray(
            dims=[tname, "latitude", "longitude"],
            coords={tname: t, "latitude": latitude, "longitude": longitude},
            attrs={"_FillValue": -999.9, "units": unit},
        )
    return xr.Dataset(var_dict)




# from stack exchange - unsure if this is linear or nearest
def interp2D_triangulation(xy, uv, d=2):
    tri = qhull.Delaunay(np.column_stack(xy))
    print("tri done")
    simplex = tri.find_simplex(np.column_stack(uv[0].flatten(), uv[1].flatten()))
    print("simplex done done")
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)
    print("bary done")
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


# from stack exchange - unsure if this is linear or nearest
def interp2D(values, vtx, wts):
    return np.einsum("nj,nj->n", np.take(values, vtx), wts)


def generate_bathy(bathy_nc, method="linear"):

    longitude, latitude, xlon, ylat, mask = load_cwm_grid()
    encoding = {"elevation": {"dtype": np.float32, "zlib": True, "complevel": 4}}
    var_dict = {
        "elevation": xr.DataArray(
            dims=["latitude", "longitude"],
            coords={"latitude": latitude, "longitude": longitude},
            attrs={"_FillValue": -999.9, "units": "m"},
        )
    }
    ds = xr.Dataset(var_dict)

    ds_in = xr.load_dataset(bathy_nc)
    lon_in = ds_in.lon[:].values
    lat_in = ds_in.lat[:].values  # input coordinates

    dude = RegularGridInterpolator((lat_in, lon_in), ds_in.elevation[...].values)
    bathy_interp = dude((ylat, xlon), method)
    ds.elevation[...] = bathy_interp

    ds.to_netcdf(
        r"X:\data\CWM\bathymetry\gebco_cwm.nc", format="netCDF4", encoding=encoding
    )
    ds.close()


def bathy_and_d2port_to_txt():

    ds = xr.load_dataset(r"X:\data\CWM\bathymetry\gebco_cwm.nc")
    np.savetxt(r"X:\data\CWM\grid\gebco_bathy_cwm.txt", ds.elevation[...].values)
    ds.close

    ds = xr.load_dataset(r"X:\data\CWM\d2port\d2port.nc")
    np.savetxt(r"X:\data\CWM\grid\d2port.txt", ds.d2p[...].values)
    ds.close


def generate_dist_2_port(input_geotiff, method="linear"):

    from osgeo import gdal, gdalconst

    # Source
    src = gdal.Open(input_geotiff, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()

    # gt the coordinates of the corners:
    width = src.RasterXSize
    height = src.RasterYSize
    gt = src.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    longitude, latitude, xlon, ylat, mask = load_cwm_grid()
    encoding = {"d2p": {"dtype": np.float32, "zlib": True, "complevel": 4}}
    var_dict = {
        "d2p": xr.DataArray(
            dims=["latitude", "longitude"],
            coords={"latitude": latitude, "longitude": longitude},
            attrs={"_FillValue": -999.9, "units": "km"},
        )
    }
    ds = xr.Dataset(var_dict)

    arr = np.array(src.GetRasterBand(1).ReadAsArray())
    lon_in = np.arange(36000) * 1 / 100 - 180 + 1 / 200
    lat_in = np.append(-1.0 * (np.arange(18000) * 1 / 100 - 90), [-90.0])
    dude = RegularGridInterpolator((np.flip(lat_in), lon_in), np.flip(arr, axis=0))
    d2p_interp = dude((ylat, xlon), method)
    ds.d2p[...] = d2p_interp

    ds.to_netcdf(r"X:\data\CWM\d2port\d2port.nc", format="netCDF4", encoding=encoding)
    ds.close()


def generate_hycom_9km(hycom_dir, year, method="linear"):
    """slow and only works for years/hycom data with -180:180 lon data orientation.
    (Exp 9X_X hycom data is 0-365 in longitude)"""
    # -- create dataset
    nc_path = os.path.join(hycom_dir, "HYCOM_%i_9km.nc" % year)
    ecco_vars = ["speed_mean", "speed_max", "speed_variance", "u_mean", "v_mean"]
    ds = create_cwm_output_dataset(year, ecco_vars, ["m/s", "m/s", "m/s", "m/s", "m/s"])
    encoding = {
        v: {
            "dtype": np.float32,
            "zlib": True,
            "complevel": 4,
            "chunksizes": (1, 360, 4320),
        }
        for v in ecco_vars
    }
    ds.to_netcdf(nc_path, format="netCDF4", encoding=encoding)
    ds.close()

    # read input coords and array sizes
    longitude, latitude, xlon, ylat, mask = load_cwm_grid()
    lon_mod = longitude
    lon_mod[
        -1
    ] = 179.92  # regular grid interp can't handle last value, somehow outside bounds
    xlon, ylat = np.meshgrid(
        lon_mod, latitude[0:2039]
    )  # constrain to input grid which ends at -80 latitude
    hypath = os.path.join(hycom_dir, "hycom_%i_z0_" % year)
    ds_in = xr.load_dataset(hypath + "001.nc")
    lon_in = ds_in.lon[:].values
    lat_in = ds_in.lat[:].values  # input coordinates
    ds_in.close()

    # re-open as netCDF4 handle, so we can sync records (can't do that w xarray?)
    rg = netCDF4.Dataset(nc_path, "a")
    ui = np.zeros(
        (2039, 4320, 64), dtype=np.float32
    )  # data array -slinging list of arrays not working
    vi = np.zeros(
        (2039, 4320, 64), dtype=np.float32
    )  # data array -slinging list of arrays not working

    for i, day in enumerate(range(1, 362, 8)):
        print("Working on week-day:", i, day)
        drange = range(day, min(day + 8, 366))
        nk = 0
        for j in drange:
            ds_in = xr.load_dataset(hypath + "%03i.nc" % j)
            print("\tLoading from: ", hypath + "%03i.nc" % j)
            for k in range(8):
                try:
                    print("\t\tInterpolating time slice: %i" % k)
                    dude = RegularGridInterpolator(
                        (lat_in, lon_in), ds_in.water_u[k, 0, ...].values
                    )
                    ui[:, :, nk] = dude((ylat, xlon), method)
                    dude = RegularGridInterpolator(
                        (lat_in, lon_in), ds_in.water_v[k, 0, ...].values
                    )
                    vi[:, :, nk] = dude((ylat, xlon), method)
                    nk += 1
                except:
                    print("\t\tTime slice not found: %i" % k)
                    continue
        print("Calculating & writing...")
        mag = np.sqrt(ui * ui + vi * vi)
        rg.variables["speed_mean"][i, 0:2039, :] = np.nanmean(mag[:, :, 0:nk], axis=2)
        rg.variables["speed_max"][i, 0:2039, :] = np.nanmax(mag[:, :, 0:nk], axis=2)
        rg.variables["speed_variance"][i, 0:2039, :] = np.nanvar(
            mag[:, :, 0:nk], axis=2
        )
        rg.variables["u_mean"][i, 0:2039, :] = np.nanmean(ui[:, :, 0:nk], axis=2)
        rg.variables["v_mean"][i, 0:2039, :] = np.nanmean(vi[:, :, 0:nk], axis=2)
        rg.sync()
    rg.close()


def generate_hycom_9km_v2(hycom_dir, year, method="linear",one_day=False,ts=False):
    """Other way was SLOW and less right. Now, find stats on hycom grid and then picks nearest from those.."""
    # -- create dataset
    if ts:
        ecco_vars = ["t_mean"]
        nc_path = os.path.join(hycom_dir, "HYCOM_ts_%i_9km.nc" % year)
        ds = create_cwm_output_dataset(year, ecco_vars, ["C",])
    else:
        if one_day:
            ecco_vars = ["speed_mean"]
            nc_path = os.path.join(hycom_dir, "HYCOM_%i_9km_1day.nc" % year)
            ds = create_cwm_1day_dataset(year, ecco_vars, ["m/s", "m/s", "m/s", "m/s", "m/s"])
        else:
            ecco_vars = ["speed_mean", "speed_max", "speed_variance", "u_mean", "v_mean"]
            nc_path = os.path.join(hycom_dir, "HYCOM_%i_9km.nc" % year)
            ds = create_cwm_output_dataset(year, ecco_vars, ["m/s", "m/s", "m/s", "m/s", "m/s"])
    encoding = {
        v: {
            "dtype": np.float32,
            "zlib": True,
            "complevel": 4,
            "chunksizes": (1, 360, 4320),
        }
        for v in ecco_vars
    }
    ds.to_netcdf(nc_path, format="netCDF4", encoding=encoding)
    ds.close()

    # read input coords and array sizes
    longitude, latitude, xlon, ylat, mask = load_cwm_grid()
    lon_mod = longitude
    lon_mod[
        -1
    ] = 179.92  # regular grid interp can't handle last value, somehow outside bounds
    xlon, ylat = np.meshgrid(
        lon_mod, latitude[0:2039]
    )  # constrain to input grid which ends at -80 latitude
    if ts:
        hypath = os.path.join(hycom_dir, "hycom_ts_%i_z0_" % year)
    else:
        hypath = os.path.join(hycom_dir, "hycom_%i_z0_" % year)
    ds_in = xr.load_dataset(hypath + "001.nc")
    lon_in = ds_in.lon[:].values
    positive_lon = False
    if np.max(lon_in) > 190:
        # grid is 0:365 instead of -180:180, we need -180:180 so will need to re-arange array below
        lon_in = lon_in - 180.0
        positive_lon = True
    lat_in = ds_in.lat[:].values  # input coordinates
    ds_in.close()

    # re-open as netCDF4 handle, so we can sync records (can't do that w xarray?)
    rg = netCDF4.Dataset(nc_path, "a")
    ui = np.zeros(
        (64, 3251, 4500), dtype=np.float32
    )  # data array -slinging list of arrays not working
    vi = np.zeros(
        (64, 3251, 4500), dtype=np.float32
    )  # data array -slinging list of arrays not working

    if one_day:
        days = range(1,366)
    else:
        days = range(1, 362, 8)

    for i, day in enumerate(days):
        
        #if i>16 and i<19:
        
            print("Working on week-day:", i, day)
            if one_day:
                drange = [day]
            else:
                drange = range(day, min(day + 8, 366))
            nk = 0
            for j in drange:
                ds_in = xr.load_dataset(hypath + "%03i.nc" % j)
                print("\tLoading from: ", hypath + "%03i.nc" % j)
                # try:
                k = ds_in.dims["time"]
                if ts:
                    ui[nk: nk + k, :, :] = np.squeeze(ds_in.water_temp[...].values)
                else:
                    ui[nk : nk + k, :, :] = np.squeeze(ds_in.water_u[...].values)
                    vi[nk : nk + k, :, :] = np.squeeze(ds_in.water_v[...].values)
                nk += k
                # except:
                #    print('\t\tTime slice not found: %i' % k)
                #    continue
            print("Calculating & writing...")
            if nk > 0:
                mag = np.sqrt(ui * ui + vi * vi)
                dude = RegularGridInterpolator(
                    (lat_in, lon_in),
                    center_on_meridian_hycom_or_not(
                        np.nanmean(mag[0:nk, :, :], axis=0), positive_lon
                    ),
                )
                if ts:
                    rg.variables["t_mean"][i, 0:2039, :] = dude((ylat, xlon), method)
                else:
                    rg.variables["speed_mean"][i, 0:2039, :] = dude((ylat, xlon), method)
                    if not one_day:
                        dude = RegularGridInterpolator(
                            (lat_in, lon_in),
                            center_on_meridian_hycom_or_not(
                                np.nanmax(mag[0:nk, :, :], axis=0), positive_lon
                            ),
                        )
                        rg.variables["speed_max"][i, 0:2039, :] = dude((ylat, xlon), method)
                        dude = RegularGridInterpolator(
                            (lat_in, lon_in),
                            center_on_meridian_hycom_or_not(
                                np.nanvar(mag[0:nk, :, :], axis=0), positive_lon
                            ),
                        )
                        rg.variables["speed_variance"][i, 0:2039, :] = dude((ylat, xlon), method)
                        dude = RegularGridInterpolator(
                            (lat_in, lon_in),
                            center_on_meridian_hycom_or_not(
                                np.nanmean(ui[0:nk, :, :], axis=0), positive_lon
                            ),
                        )
                        rg.variables["u_mean"][i, 0:2039, :] = dude((ylat, xlon), method)
                        dude = RegularGridInterpolator(
                            (lat_in, lon_in),
                            center_on_meridian_hycom_or_not(
                                np.nanmean(vi[0:nk, :, :], axis=0), positive_lon
                            ),
                        )
                        rg.variables["v_mean"][i, 0:2039, :] = dude((ylat, xlon), method)
                    else:
                        # should maybe only happen with one-day, all 8-day years calculated OK, holes filled
                        rg.variables["speed_mean"][i, ...] = rg.variables["speed_mean"][i-1, ...]
    
            rg.sync()
            
    rg.close()


def center_on_meridian(data_grid):
    datai = np.full(np.shape(data_grid), np.nan)
    datai[:, :360] = data_grid[:, 360:]
    datai[:, 360:] = data_grid[:, :360]
    return datai


def center_on_meridian_hycom_or_not(data_grid, recenter=True):
    if recenter:
        datai = np.full(np.shape(data_grid), np.nan, np.float32)
        datai[:, :2250] = data_grid[:, 2250:]
        datai[:, 2250:] = data_grid[:, :2250]
        return datai
    else:
        return data_grid


def mask_outside_radius(x, y, data_grid_i, xi_grid, yi_grid):
    nx, ny = np.shape(data_grid_i)
    for i in range(nx):
        print("\t\t%i of %i" % (i, nx))
        for j in range(nx):
            dx = np.min(np.abs(x - xi_grid[i, j]))
            dy = np.min(np.abs(y - yi_grid[i, j]))
            if max(dx, dy) > 0.5:
                data_grid_i[i, j] = np.nan
    return data_grid_i


def generate_ecmwf_9km(ecmwf_dir, year, ec_vars=["swh", "mwp"], method="nearest", one_day=False):
    # -- create dataset
    ec_units = ["m", "s"]
    cwm_vars = []
    cwm_units = []
    for i, v in enumerate(ec_vars):
        cwm_vars += ["%s_mean" % v, "%s_max" % v, "%s_variance" % v]
        cwm_units += [ec_units[i], ec_units[i], ec_units[i]]
        if one_day:
            nc_path = os.path.join(ecmwf_dir, "ecmwf_cwm_%i_9km_1day.nc" % year)
        else:
            nc_path = os.path.join(ecmwf_dir, "ecmwf_cwm_%i_9km.nc" % year)
    if not os.path.isfile(nc_path):
        if one_day:
            create_cwm_1day_dataset_netCDF4(year, cwm_vars, cwm_units, nc_path, close=True)
        else:
            ds = create_cwm_output_dataset(year, cwm_vars, cwm_units)
            encoding = {
                v: {
                    "dtype": np.float32,
                    "zlib": True,
                    "complevel": 4,
                    "chunksizes": (1, 360, 4320),
                }
                for v in cwm_vars
            }
            ds.to_netcdf(nc_path, format="netCDF4", encoding=encoding)
            ds.close()

    # read input coords and array sizes
    longitude, latitude, xlon, ylat, mask = load_cwm_grid()

    # open input NC3 file
    ds_in = [
        xr.load_dataset(os.path.join(ecmwf_dir, "%s_%i.nc" % (v, year)))
        for v in ec_vars
    ]
    lati = ds_in[0].latitude.values
    loni = (
        ds_in[0].longitude.values + 0.25 - 180.0
    )  # we shift data by 180 degrees below
    loni[
        0
    ] = (
        -180.0
    )  # expand outer values so nearest regular grid interpolation works at edges
    loni[
        -1
    ] = 180.0  # expand outer values so nearest regular grid interpolation works at edges

    # re-open as netCDF4 handle, so we can sync records (can't do that w xarray?)
    rg = netCDF4.Dataset(nc_path, "a")
    if one_day:
        days = range(1,366)
    else:
        days = range(1, 362, 8)
    
    for i, day in enumerate(days):
        print("Working on week-day:", i, day)
        st = (day - 1) * 24
        if one_day:
            end = min(st+24, 8759)
        else:
            end = min(st + 8 * 24, 8759)
        for j, v in enumerate(ec_vars):
            data = ds_in[j][v][st:end, :, :].values
            print("\tProcessing mean...%s" % v)
            data_mean = center_on_meridian(np.nanmean(data, axis=0))
            dude = RegularGridInterpolator(
                (np.flip(lati), loni), np.flip(data_mean, axis=0)
            )  # handles nans
            rg.variables["%s_mean" % v][i, :, :] = dude((ylat, xlon), method)
            print("\tProcessing max...%s" % v)
            data_max = center_on_meridian(np.nanmax(data, axis=0))
            dude = RegularGridInterpolator(
                (np.flip(lati), loni), np.flip(data_max, axis=0)
            )
            rg.variables["%s_max" % v][i, :, :] = dude((ylat, xlon), method)
            print("\tProcessing variance...%s" % v)
            data_var = center_on_meridian(np.nanvar(data, axis=0))
            dude = RegularGridInterpolator(
                (np.flip(lati), loni), np.flip(data_var, axis=0)
            )
            rg.variables["%s_variance" % v][i, :, :] = dude((ylat, xlon), method)
        rg.sync()
    rg.close()


def generate_fseq(fseq_mat_file, out_nc_fname, method="nearest"):

    # read input coords and array sizes
    with h5py.File(fseq_mat_file, "r") as fp:
        blon = fp["blon"][...]
        blat = fp["blat"][...]
        fseq = fp["fseq_cwm"][...]
    blon1 = blon[:, 0]
    blat1 = blat[0, :]

    blon1_180 = blon1 - 180
    blon1_180[
        0
    ] = (
        -180.0
    )  # expand outer values so nearest regular grid interpolation works at edges
    blon1_180[
        -1
    ] = 180.0  # expand outer values so nearest regular grid interpolation works at edges
    blat1[0] = -90.0
    blat1[-1] = 90.0

    longitude, latitude, xlon, ylat, mask = load_cwm_grid()

    # center_on_meridian(data_grid):
    datai = np.full(np.shape(fseq), np.nan)
    datai[:90, :] = fseq[90:, :]
    datai[90:, :] = fseq[:90, :]

    dude = RegularGridInterpolator(
        (blat1, blon1_180), np.transpose(datai)
    )  # handles nans
    # dude = RegularGridInterpolator((np.flip(lati), loni), np.flip(data_mean,axis=0)) # handles nans
    fseq_cwm = dude((ylat, xlon), method)
    create_write_cwm_annual_dataset(
        ["fseq_bottom_500y"], ["fractional"], [fseq_cwm], out_nc_fname, dtype_list=None
    )


def generate_fseq_10(fseq_mat_pathstub, out_nc_fname, method="nearest"):

    for y in range(100, 1001, 100):

        fseq_mat_file = fseq_mat_pathstub + "%i" % y + ".mat"

        # read input coords and array sizes
        with h5py.File(fseq_mat_file, "r") as fp:
            if y == 100:
                blon = fp["blon"][...]
                blat = fp["blat"][...]
            fseq = fp["fseq_cwm"][...]

        if y == 100:
            blon1 = blon[:, 0]
            blat1 = blat[0, :]

            blon1_180 = blon1 - 180
            blon1_180[
                0
            ] = (
                -180.0
            )  # expand outer values so nearest regular grid interpolation works at edges
            blon1_180[
                -1
            ] = 180.0  # expand outer values so nearest regular grid interpolation works at edges
            blat1[0] = -90.0
            blat1[-1] = 90.0

            longitude, latitude, xlon, ylat, mask = load_cwm_grid()

            varnames = ["fseq_bottom_%iy" % ii for ii in range(100, 1001, 100)]
            units = ["fractional" for ii in range(10)]
            ds = create_cwm_annual_dataset(varnames, units)

        # center_on_meridian(data_grid):
        datai = np.full(np.shape(fseq), np.nan)
        datai[:90, :] = fseq[90:, :]
        datai[90:, :] = fseq[:90, :]

        dude = RegularGridInterpolator(
            (blat1, blon1_180), np.transpose(datai)
        )  # handles nans
        # dude = RegularGridInterpolator((np.flip(lati), loni), np.flip(data_mean,axis=0)) # handles nans

        vname = "fseq_bottom_%iy" % y
        f = dude((ylat, xlon), method)
        f[f < 0.0] = 0.0
        f[f > 1.0] = 1.0
        ds[vname][...] = f
        print("generated year %i..." % y)

    encoding = {
        v: {"dtype": np.float32, "zlib": True, "complevel": 4} for v in varnames
    }
    ds.to_netcdf(out_nc_fname, format="netCDF4", encoding=encoding)


def generate_ecco_9km(ecco_dir, year, method="cubic"):
    # create dataset
    ecco_vars = ["speed_mean", "speed_max", "speed_variance", "u_mean", "v_mean"]
    ds = create_cwm_output_dataset(year, ecco_vars, ["m/s", "m/s", "m/s", "m/s", "m/s"])
    encoding = {
        v: {
            "dtype": np.float32,
            "zlib": True,
            "complevel": 4,
            "chunksizes": (1, 360, 4320),
        }
        for v in ecco_vars
    }
    nc_path = os.path.join(ecco_dir, "ECCO_%i_9km.nc" % year)
    ds.to_netcdf(nc_path, format="netCDF4", encoding=encoding)
    longitude, latitude, xlon, ylat, mask = load_cwm_grid()

    # read input coords and array sizes
    upath = os.path.join(ecco_dir, "EVEL_%i_" % year)
    vpath = os.path.join(ecco_dir, "NVEL_%i_" % year)
    ds_in = xr.load_dataset(upath + "001.nc")
    lonin = ds_in.XC.values.flatten()
    latin = ds_in.YC.values.flatten()

    # main loop - unclear if enough memory for big xarray dataset?  Can we use float32?
    # re-open as netCDF4 handle, so we can sync records (can't do that w xarray?)
    rg = netCDF4.Dataset(nc_path, "a")
    for i, day in enumerate(range(1, 362, 8)):
        print("working on week-day:", i, day)
        drange = range(day, min(day + 8, 366))
        ui = []
        vi = []
        for j in drange:
            u = (
                xr.load_dataset(upath + "%03i.nc" % j).EVEL[0, 0, ...].values.flatten()
            )  # 0=time,0=k_level [5, 15, 25 ... ] m
            valid = np.isfinite(u)
            ui.append(
                griddata(
                    (lonin[valid], latin[valid]), u[valid], (xlon, ylat), method=method
                )
            )
            v = xr.load_dataset(vpath + "%03i.nc" % j).NVEL[0, 0, ...].values.flatten()
            valid = np.isfinite(v)
            vi.append(
                griddata(
                    (lonin[valid], latin[valid]), v[valid], (xlon, ylat), method=method
                )
            )
        ui = np.dstack(ui)
        vi = np.dstack(vi)
        mag = np.sqrt(ui * ui + vi * vi)
        rg.variables["speed_mean"][i, ...] = do_mask(np.nanmean(mag, axis=2), mask)
        rg.variables["speed_max"][i, ...] = do_mask(np.nanmax(mag, axis=2), mask)
        rg.variables["speed_variance"][i, ...] = do_mask(np.nanvar(mag, axis=2), mask)
        rg.variables["u_mean"][i, ...] = do_mask(np.nanmean(ui, axis=2), mask)
        rg.variables["v_mean"][i, ...] = do_mask(np.nanmean(vi, axis=2), mask)
        rg.sync()
    rg.close()


def do_mask(arr, mask):
    arr[mask] = np.nan
    return arr


def compare_woa_scatter():
    import random

    ds_woa = xr.load_dataset(
        "/Volumes/Aux/more_data/CWM/WOA_N/WOA_N_8day_interp_9km_cubic.nc"
    )
    ds = xr.load_dataset("/Volumes/Aux/more_data/CWM/nitrogen/N_static_2003.nc")

    # for i,day in enumerate(eight_day_weeks()):
    #    ds_WOA =

    idx = random.sample(range(4320 * 2160), 1000)


def dl_woa18():
    import urllib.request

    for i in range(1, 13):
        print("Downloading: %s" % ("woa18_decav_s%02i_01.nc" % i))
        urllib.request.urlretrieve(
            r"https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/salinity/decav/1.00/woa18_decav_s"
            + "%02i_01.nc" % i,
            "woa18_decav_s%02i_01.nc" % i,
        )
        urllib.request.urlretrieve(
            r"https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/salinity/A5B7/1.00/woa18_A5B7_s"
            + "%02i_01.nc" % i,
            "woa18_A5B7_s%02i_01.nc" % i,
        )
        urllib.request.urlretrieve(
            r"https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/nitrate/all/1.00/woa18_all_n"
            + "%02i_01.nc" % i,
            "woa18_all_n%02i_01.nc" % i,
        )


def dl_behrenfeld():
    import urllib.request

    # for i in range(2013,2019):
    #     print('Downloading PAR %i'%i)
    #     urllib.request.urlretrieve(r'http://orca.science.oregonstate.edu/data/2x4/8day/par.modis.r2018/hdf/par.m.'+'%i.tar'%i,'par.m.'+'%i.tar'%i)
    #
    # for i in range(2003,2019):
    #     if i not in [2003,2010,2019]:
    #         print('Downloading SST %i'%i)
    #         urllib.request.urlretrieve(r'http://orca.science.oregonstate.edu/data/2x4/8day/sst.modis.r2018/hdf/sst.m.'+'%i.tar'%i,'sst.m.'+'%i.tar'%i)
    #
    # for i in range(2003,2019):
    #     if i not in [2003,2010,2019]:
    #         print('Downloading cbpm %i'%i)
    #         urllib.request.urlretrieve(r'http://orca.science.oregonstate.edu/data/2x4/8day/cbpm2.modis.r2018/hdf/cbpm.m.'+'%i.tar'%i,'cbpm.m.'+'%i.tar'%i)

    for i in range(2003, 2019):
        if i not in [2003, 2010, 2019]:
            print("Downloading MLD %i" % i)
            urllib.request.urlretrieve(
                r"http://orca.science.oregonstate.edu/data/2x4/8day/mld125.hycom/hdf/mld.hycom_125."
                + "%i.tar" % i,
                "mld.hycom_125." + "%i.tar" % i,
            )

    for i in range(2003, 2019):
        if i not in [2003, 2010, 2019]:
            print("Downloading CHL %i" % i)
            urllib.request.urlretrieve(
                r"http://orca.science.oregonstate.edu/data/2x4/8day/chl.modis.r2018/hdf/chl.m."
                + "%i.tar" % i,
                "chl.m." + "%i.tar" % i,
            )


def get_output_slice(output_file, var_name, time_slice_index=None):
    if output_file.endswith(".mat") or output_file.endswith(".h5"):
        fp = h5py.File(output_file, "r")
        if "output" in fp.keys():
            dset = fp["output"][var_name]
        else:
            dset = fp[var_name]
        if time_slice_index is None:
            return dset[...]
        else:
            return dset[time_slice_index, ...]
    else:
        nc = netCDF4.Dataset(output_file, "r")
        if time_slice_index is None:
            return nc.variables[var_name][...].filled(np.nan)
        else:
            return nc.variables[var_name][time_slice_index, ...].filled(np.nan)


def mask_ravel(arr,mask):
    arr_masked = arr[mask].ravel()
    


def cwm_results_to_csv(nc_file_var_list, mask, csv_fname):
    """Takes a list of (nc_filepath,nc_varname) tuples, whish describe the
    netcdf file containing a CWM results grid, and the variable we want to
    convert to text-based CSV. The resulting CSV data will be masked by
    "mask" since I can't imagine you would want to output the entire grid
    to CSV - very big."""

    # longitude,latitude,xlon,ylat = load_cwm_grid_no_mask()
    longitude = np.arange(4320) * 1 / 12 - 180 + 1 / 24
    latitude = -1.0 * (np.arange(2160) * 1 / 12 - 90 + 1 / 24)
    xlon, ylat = np.meshgrid(longitude, latitude)  # output coordinate meshes

    df_info = {'longitude':xlon[mask].ravel(),'latitude':ylat[mask].ravel()} 
    for (ncf,ncvar) in nc_file_var_list:
        ds = xr.open_dataset(ncf)
        df_info[ncvar] = ds[ncvar].values[mask].ravel()

    pd.Dataset(df_info).to_csv(csv_fname)


def generate_mask_from_SST():
    
    base_path = r'X:/data/CWM/SST/sst_'
    
    mask = np.full((2160,4320),np.nan)
    
    for y in range(2003,2020):
        nc = netCDF4.Dataset(base_path + '%4i.nc'%y)
        for i in range(46):
            print(y,i)
            #mask = np.nansum(mask,nc.variables['sst'][i,...].filled(np.nan).squeeze())
            mask = nansum_old(np.dstack((mask, nc.variables['sst'][i,...].filled(np.nan).squeeze())), 2)
        nc.close()
    
    mask[np.isfinite(mask)] = 1
    mask[np.isnan(mask)] = 0
    mask = mask.astype(np.int32) #    mask = np.isfinite(mask).astype(np.int32)
    create_write_cwm_annual_dataset(['mask'], ['0to1'], [mask], 'X:/data/CWM/regions_and_masks/mask_from_sst.nc', [np.float32])   
    np.savetxt('X:/data/CWM/regions_and_masks/mask_from_sst.txt', mask, fmt="%1i")
    

def test_plot_nc(nc_file, nc_var):
    nc = netCDF4.Dataset(nc_file, "r")
    dims = nc.variables[nc_var].shape
    if len(dims) == 2:
        data = nc.variables[nc_var][...].filled(np.nan)
    else:
        data = nc.variables[nc_var][0, ...].filled(np.nan)
        for i in range(1, dims[0]):
            data += nc.variables[nc_var][i, ...].filled(np.nan)
    nc.close()
    plt.figure(figsize=[8,4])
    plt.imshow(data)
    plt.colorbar()

    plt.savefig(nc_file[:-3]+".png",dpi=800,bbox_inches="tight")

    #plt.show()


def interpolate_chl_nc(cwm_chla_nc,year):

    interp_name = cwm_chla_nc[:-2] + 'interp.nc'
    longitude, latitude, xlon, ylat, cwmmask = load_cwm_grid()
    
    
    # ds = create_cwm_output_dataset(year, ['chl'], ['mg chla m-3'])
    # encoding = {
    #     'chl': {
    #         "dtype": np.float32,
    #         "zlib": True,
    #         "complevel": 4,
    #         "chunksizes": (1, 360, 4320),
    #     }
    # }
    # ds.to_netcdf(interp_name, format="netCDF4", encoding=encoding)

    rg_in = netCDF4.Dataset(cwm_chla_nc, "r")
    rg = netCDF4.Dataset(interp_name, "a")
    dims = rg_in.variables['chl'].shape
    x = np.arange(0, dims[2])
    y = np.arange(0, dims[1])
    xx, yy = np.meshgrid(x, y)
    for i in range(0, 1): #dims[0]):            

        print('Interpolating array %i of %i...'%(i,dims[0]))        

        #mask invalid values
        array = rg_in.variables['chl'][i,...].filled(np.nan).squeeze()  #np.ma.masked_invalid(array)
        #get only the valid values
        mask = np.isfinite(array)
        x1 = xx[mask].squeeze()
        y1 = yy[mask].squeeze()
        newarr = array[mask].squeeze()

        interp_arr = griddata((x1, y1), newarr.ravel(),
                                  (xx, yy), method='cubic')
        interp_arr[cwmmask] = np.nan
        
        rg.variables['chl'][i,...] = interp_arr
        rg.sync()
    
    rg_in.close()
    rg.close()    


def median_filter_nc_append(nc_file,var,ksize=3):
    from scipy.ndimage import median_filter
    nc = netCDF4.Dataset(nc_file,'r')
    arr_mf = median_filter(nc.variables[var][...],ksize)
    create_write_cwm_annual_dataset(
        [var], [nc.variables[var].units], [arr_mf], 
        nc_file[:-3]+'_mf%i.nc'%ksize, 
        dtype_list=[np.int32]
    )
    nc.close()


def mode_filter_nc_append(nc_file,var,ksize=3):
    #from scipy.ndimage import median_filter
    k = int((ksize-1)/2)    
    nc = netCDF4.Dataset(nc_file,'r')
    arr_pad = np.pad(nc.variables[var][...],((k,),(k,)),mode='reflect')
    arr_mf = mode_filter(arr_pad,ksize)
    create_write_cwm_annual_dataset(
        [var], [nc.variables[var].units], [arr_mf], 
        nc_file[:-3]+'_modef%i.nc'%ksize, 
        dtype_list=[np.int32]
    )
    nc.close()

@jit(nopython=True)
def mode_filter(arr,filter_size):
    nx,ny = np.shape(arr)
    k = int((filter_size-1)/2)
    mode_out = np.zeros((nx-2*k,ny-2*k),np.int32)
    for i in range(k,nx-k):
        for j in range(k,ny-k):
            mode_out[i-k,j-k] = np.argmax(np.bincount(arr[i-k:i+k,j-k:j+k].ravel()))
    return mode_out


def regrid_hycom_to_m180(hycom0_fname,new_fname,hycom_m180_example):
    
    ds_ex = xr.open_dataset(hycom_m180_example,engine='netcdf4')
    if '_ts_' in hycom_m180_example:
        regrid_vars = ['water_temp','salinity']
    else:
        regrid_vars =  ['water_u','water_v']

    ds_out = xr.open_dataset(hycom0_fname,engine='netcdf4')
    nt = ds_out.dims["time"]
    for i in range(nt): 
        for var in regrid_vars:
            ds_out[var][i,0,...] = center_on_meridian_hycom_or_not(ds_out[var][i,0,...], True)
    ds_out['lon'] = ds_ex['lon']
    ds_out.to_netcdf(new_fname)


def cwm_coast_interp(cwm_nc_input,new_mask):
    
    pass



def filter_bad_winter_par(par_data,ylat,d8_index,bad_value=0.1,threshold=10.):
    # NH winter/spring
    if d8_index < 3:
        par_data[np.logical_and(ylat > 62.,par_data > threshold)] = bad_value
    elif d8_index < 6:
        par_data[np.logical_and(ylat > 66.,par_data > threshold)] = bad_value
    elif d8_index < 9:
        par_data[np.logical_and(ylat > 75.,par_data > threshold)] = bad_value

    # SH fall/winter/spring
    elif d8_index >= 14 and d8_index < 17:
        par_data[np.logical_and(ylat < -62.,par_data > threshold)] = bad_value
    elif d8_index >= 17 and d8_index < 20:
        par_data[np.logical_and(ylat < -57.,par_data > threshold)] = bad_value
    elif d8_index >= 20 and d8_index < 23:
        par_data[np.logical_and(ylat < -53.,par_data > threshold)] = bad_value
    elif d8_index >= 23 and d8_index < 26:
        par_data[np.logical_and(ylat < -57.,par_data > threshold)] = bad_value
    elif d8_index >= 26 and d8_index < 29:
        par_data[np.logical_and(ylat < -62.,par_data > threshold)] = bad_value

    # NH fall/winter
    elif d8_index >= 35 and d8_index < 38:
        par_data[np.logical_and(ylat > 70.,par_data > threshold)] = bad_value
    elif d8_index >= 38 and d8_index < 41:
        par_data[np.logical_and(ylat > 62.,par_data > threshold)] = bad_value
    elif d8_index >= 41:
        par_data[np.logical_and(ylat > 60.,par_data > threshold)] = bad_value
    return par_data


#@jit(nopython=True)
def horizontal_interp(nlat,nlon,cwm_data,mask,latitude,max_lat=80,min_lat=-80):
    x = np.arange(nlon)
    for i in range(nlat):
        if latitude[i] < max_lat and latitude[i] > min_lat:
            data = cwm_data[i,:]
            valid = np.isfinite(data)
            missing = np.logical_and(~valid,mask[i,:])
            if np.sum(valid) > 100 and np.sum(missing) > 0:
                # -- can't get this to work with numba
                # fill_data = np.interp(x[missing],x[valid],data[valid])#,left=np.nan,right=np.nan)
                # ms = np.nonzero(missing)[0]
                # for j in ms:
                #     cwm_data[i,j] = fill_data[j]
                fill_data = np.interp(x[missing], x[valid], data[valid],left=np.nan,right=np.nan)
                cwm_data[i, missing] = fill_data
    return cwm_data


def interpolate_missing_cwm(file_path, ncvars, years, mask_fn=default_cwm_mask, lat_interp=True, lat_filter=0.1):
    longitude, latitude, xlon, ylat, grid_mask = load_cwm_grid(mask_fn=mask_fn)
    nlon=len(longitude)
    nlat=len(latitude)
    grid_mask = ~grid_mask

    ll_in = np.vstack([xlon.ravel(),ylat.ravel()]).T

    for year in years:
        fn = file_path.replace('YYYY', '%i' % year)
        ds_in = xr.open_dataset(file_path.replace('YYYY', '%i' % year))

        nt, _, _ = np.shape(ds_in[ncvars[-1]])

        #for t in [0,1]:
        for t in range(nt):
            kd_cache = None
            for varname in ncvars:
                data_map = ds_in[varname][t,...].values
                print(varname,year,' -- ',t)
                if lat_filter is not None:
                    data_map = filter_bad_winter_par(data_map,ylat,t,bad_value=lat_filter,threshold=5.)
                if lat_interp:
                    data_map = horizontal_interp(nlat,nlon,data_map,grid_mask,latitude,max_lat=62.,min_lat=-60)
                else:
                    valid = np.isfinite(data_map)
                    missing = np.logical_and(grid_mask,~valid)
                    valid_r = valid.ravel()
                    missing_r = missing.ravel()
                    #data_map[missing] = spint.griddata((xlon[valid], ylat[valid]),
                    #                                 data_map[valid],
                    #                                 (xlon[missing], ylat[missing]),
                    #                                 method='nearest', fill_value=np.nan)

                    data_map[missing],_ = interpolate_nearest_w_max_dist(ll_in[valid_r], data_map[valid].ravel(),
                                                                                ll_in[missing_r], kd_cache=kd_cache, max_dist=1.0)

                ds_in[varname][t,...] = data_map
        ds_in_new = ds_in[ncvars]
        ds_in_new.to_netcdf(fn[:-3] + '_gridinterp.nc')


def interpolate_missing_cwm_nc4(file_path, ncvars, years, mask_fn=default_cwm_mask, lat_interp=True, lat_filter=0.1):
    longitude, latitude, xlon, ylat, grid_mask = load_cwm_grid(mask_fn=mask_fn)
    nlon=len(longitude)
    nlat=len(latitude)
    grid_mask = ~grid_mask

    ll_in = np.vstack([xlon.ravel(),ylat.ravel()]).T

    for year in years:
        fn = file_path.replace('YYYY', '%i' % year)
        fn_new = fn[:-3] + '_gridinterp_v2.nc'

        # copy old nc, open as 'a'
        #shutil.copyfile(fn, fn_new)
        nc_in = netCDF4.Dataset(fn,'r')
        nc_out = netCDF4.Dataset(fn_new,'a')

        nt = nc_in.dimensions['time'].size

        #for t in [0,1]:
        for t in range(nt):
            kd_cache = None
            for varname in ncvars:
                data_map = nc_in.variables[varname][t,...]
                print(varname,year,' -- ',t)
                if lat_filter is not None:
                    data_map = filter_bad_winter_par(data_map,ylat,t,bad_value=lat_filter,threshold=5.)
                if lat_interp:
                    data_map = horizontal_interp(nlat,nlon,data_map,grid_mask,latitude,max_lat=62.,min_lat=-60)
                else:
                    valid = np.isfinite(data_map)
                    missing = np.logical_and(grid_mask,~valid)
                    valid_r = valid.ravel()
                    missing_r = missing.ravel()
                    #data_map[missing] = spint.griddata((xlon[valid], ylat[valid]),
                    #                                 data_map[valid],
                    #                                 (xlon[missing], ylat[missing]),
                    #                                 method='nearest', fill_value=np.nan)

                    data_map[missing],_ = interpolate_nearest_w_max_dist(ll_in[valid_r], data_map[valid].ravel(),
                                                                                ll_in[missing_r], kd_cache=kd_cache, max_dist=1.0)

                nc_out.variables[varname][t,...] = data_map
                nc_out.sync()
        nc_in.close()
        nc_out.close()




# this is not finished ....
def interpolate_nearest_w_max_dist(ll_in,data_in,ll_out,kd_cache=None,max_dist=None):
    if kd_cache is not None:
        kdt,dist,ind,data_mask = kd_cache
    else:
        data_mask = np.isfinite(data_in)
        kdt = spatial.cKDTree(ll_in[data_mask],copy_data=True)
        dist,ind = kdt.query(ll_out)
    data_interp = data_in[data_mask][ind]
    if max_dist is not None:
        data_interp[dist > max_dist] = np.nan
    return data_interp, (kdt,dist,ind,data_mask)

@jit(nopython=True)
def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8 # 6372.8 km radius of earth

    dLat = np.deg2rad(lat2 - lat1)
    dLon = np.deg2rad(lon2 - lon1)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    a = math.sin(dLat / 2.)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2.)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

@jit(nopython=True)
def gen_coast_masks(mask,ylat):
    coast_mask_9 = np.copy(mask)
    coast_mask_18 = np.copy(mask)
    coast_mask_28 = np.copy(mask)
    coast_mask_56 = np.copy(mask)
    coast_mask_102 = np.copy(mask)

    lon_km_equator = haversine(0, 0, 0, 1);

    for i in range(2130): # should this be 2160???
        clat = ylat[i, 1]
        if np.abs(clat) < 80.0:
            d_ratio = lon_km_equator / haversine(clat, 0, clat, 1);
            n9 = round(d_ratio)
            n18 = round(2 * d_ratio)
            n28 = round(3 * d_ratio)
            n56 = round(6 * d_ratio)
            n102 = round(11 * d_ratio)
            for j in range(4320):
                if mask[i, j] == 0:
                    coast_mask_9[max(1, i - 1): min(2130, i + 1), max(1, j - n9): min(4320, j + n9)] = 0
                    coast_mask_18[max(1, i - 2): min(2130, i + 2), max(1, j - n18): min(4320, j + n18)] = 0
                    coast_mask_28[max(1, i - 3): min(2130, i + 3), max(1, j - n28): min(4320, j + n28)] = 0
                    coast_mask_56[max(1, i - 6): min(2130, i + 6), max(1, j - n56): min(4320, j + n56)] = 0
                    coast_mask_102[max(1, i - 11): min(2130, i + 11), max(1, j - n102): min(4320, j + n102)] = 0
    coast_102 = ~(coast_mask_102 - mask)
    coast_56 = ~(coast_mask_56 - mask)
    coast_28 = ~(coast_mask_28 - mask)
    coast_18 = ~(coast_mask_18 - mask)
    coast_9 = ~(coast_mask_9 - mask)
    return coast_9,coast_18,coast_28,coast_56,coast_102

def gen_bathy_masks(bathy,mask):
    shelf_mask_100 = np.logical_and(bathy >= -100.0, bathy <= 0.0)
    shelf_mask_100 = np.logical_and(mask, shelf_mask_100)
    shelf_mask_150 = np.logical_and(bathy >= -150.0, bathy <= 0.0)
    shelf_mask_150 = np.logical_and(mask, shelf_mask_150)
    shelf_mask_200 = np.logical_and(bathy >= -200.0, bathy <= 0.0)
    shelf_mask_200 = np.logical_and(mask, shelf_mask_200)
    return shelf_mask_100,shelf_mask_150,shelf_mask_200


def get_bathy(bathy_nc = r'X:\data\CWM\bathymetry\gebco_cwm.nc'):
    ds = xr.open_dataset(bathy_nc)
    bathy = ds['elevation'][...].values
    ds.close()
    return bathy


def save_new_analysis_masks(mask_fn,nc_file):
    longitude, latitude, xlon, ylat, grid_mask = load_cwm_grid(mask_fn=mask_fn)
    grid_mask = ~grid_mask
    bathy = get_bathy()
    coasts = gen_coast_masks(grid_mask, ylat)
    shelfs = gen_bathy_masks(bathy,grid_mask)
    var_list_coast = ['coast_9','coast_18','coast_28','coast_56','coast_102']
    var_list_shelf = ['shelf_mask_100','shelf_mask_150','shelf_mask_200']
    ds = xr.Dataset()
    ds['latitude'] = ('latitude'),latitude
    ds['longitude'] = ('longitude'),longitude
    for i,v in enumerate(var_list_coast):
        ds[v] = ('latitude','longitude'),coasts[i]
    for i,v in enumerate(var_list_shelf):
        ds[v] = ('latitude','longitude'),shelfs[i]
    ds['mask'] = ('latitude','longitude'),grid_mask
    ds.to_netcdf(nc_file)

def save_carib_gom_masks(prev_mask_nc=r'X:\data\CWM\regions_and_masks\analysis_masks_20220412.nc',
                         out_nc_path=r'X:\data\CWM\regions_and_masks\analysis_masks_20220608.nc'):
    import fiona

    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()

    mask_files = {
        'Caribbean': r'X:\data\CWM\regions_and_masks\Caribbean.shp',
        'Gulf_of_Mexico': r'X:\data\CWM\regions_and_masks\Gulf_of_Mexico.shp',
    }

    outmasks = {}
    for name,f in mask_files.items():
        fc = fiona.open(f)
        for feature in fc:
            geom = feature["geometry"]
            coord_set = geom["coordinates"]
            m1 = np.full(np.shape(xlon), False, dtype=bool)
            for poly in coord_set:
                coords = np.asarray(poly)
                m1 = np.logical_or(
                    m1, is_inside_sm_parallel(xlon, ylat, coords)
                )
            outmasks[name] = m1

    ds = xr.open_dataset(prev_mask_nc)
    for name in mask_files.keys():
        ds[name] = ('latitude', 'longitude'), outmasks[name]

    encoding = {}
    for varname, da in ds.data_vars.items():
        if len(ds[varname].dims) > 1:
            encoding[varname] = {"dtype": "i1", "zlib": True, "complevel": 4}

    ds.to_netcdf(out_nc_path,format='netCDF4',encoding=encoding)


def generate_ecco_darwin_v5_cwm(data_dir,grid_dir,out_nc,year=2017):

    import xmitgcm,pyresample

    varname = 'NO3_20m'
    # load mds
    ds = xmitgcm.open_mdsdataset(data_dir,grid_dir,geometry='llc')#,iters=[705672,707832])

    # Extract LLC 2D coordinates
    lons_1d = ds.XC.values.ravel()
    lats_1d = ds.YC.values.ravel()

    # Define original grid
    orig_grid = pyresample.geometry.SwathDefinition(lons=lons_1d, lats=lats_1d)

    # Longitudes latitudes to which we will we interpolate
    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()
    new_grid = pyresample.geometry.GridDefinition(lons=xlon,lats=ylat)

    cwm_data = create_cwm_monthly_dataset(year,[varname],['mmol/m^3'])
    for i in range(12):
        data_20m = ds['TRAC02'][i,0:1,...].values
        data_20m = np.nanmean(data_20m,axis=0)
        cwm_data[varname][i,...] = pyresample.kd_tree.resample_nearest(orig_grid, data_20m,new_grid,
                                            radius_of_influence=100000,
                                            fill_value=None)
        #im1 = plt.imshow(cwm_data[varname][i,...],vmin=0,vmax=10)
        #plt.colorbar(im1)
        #plt.show()

    encoding = {varname: {"dtype": np.float32, "zlib": True, "complevel": 4}}
    cwm_data.to_netcdf(out_nc, format='netCDF4', encoding=encoding)

def make_best_spp_nc(eucheuma_output_nc,out_nc_filepath,replace_nan=0.0,mc_results=False):
    spp_order = ['Eucheuma','Sargassum','Porphyra','Saccharina','Macrocystis']
    harv = np.full([5,2160,4320],np.nan,np.float32)
    for i,spp, in enumerate(spp_order):
        ds = xr.open_dataset(eucheuma_output_nc.replace('Eucheuma',spp),decode_times=False)
        print('Reading:',eucheuma_output_nc.replace('Eucheuma',spp))
        if mc_results:
            harv[i, ...] = ds['harv_median'].values
        else:
            harv[i,...] = ds['harv_sum'].values
        ds.close()
    
    if replace_nan is not None:
        np.nan_to_num(harv,copy=False,nan=replace_nan)    
    index_H = np.nanargmax(harv,axis=0)

    Harvest = np.nanmax(harv,axis=0)
    del(harv)

    nharv = np.zeros([5,2160,4320],np.int32)
    for i,spp, in enumerate(spp_order):
        ds = xr.open_dataset(eucheuma_output_nc.replace('Eucheuma',spp),decode_times=False)
        print('Reading:',eucheuma_output_nc.replace('Eucheuma',spp))
        if mc_results:
            nharv[i, ...] = ds['n_harv_median'].values
        else:
            nharv[i,...] = ds['n_harv_sum'].values
        ds.close()
    nharv_H = index_H.choose(nharv) #nharv[index_H,:,:]  -- choose seems weird.
    del(nharv)

    grow = np.zeros([5,2160,4320],np.int32)
    for i,spp, in enumerate(spp_order):
        ds = xr.open_dataset(eucheuma_output_nc.replace('Eucheuma',spp),decode_times=False)
        print('Reading:',eucheuma_output_nc.replace('Eucheuma',spp))
        if mc_results:
            grow[i, ...] = ds['Growth2_median'].values
        else:
            grow[i,...] = ds['Growth2_sum'].values
        ds.close()

    if replace_nan is not None:
        np.nan_to_num(grow,copy=False,nan=replace_nan)    
   
    Growth = index_H.choose(grow) #nharv[index_H,:,:]  -- choose seems weird.
    del(grow)

    (
        longitude,
        latitude,
        xlon,
        ylat,
        mask,
        area,
        lonb,
        latb,
        xlonb,
        ylatb,
    ) = load_cwm_grid_area_bounds()

    ds = xr.Dataset()
    ds['latitude'] = ('latitude'),latitude
    ds['latitude'].attrs['units'] = 'Degrees N'
    ds['latitude_bounds'] = ('latitude_bounds'),latb
    ds['latitude_bounds'].attrs['units'] = 'Degrees N'

    ds['longitude'] = ('longitude'),longitude
    ds['longitude'].attrs['units'] = 'Degrees E'
    ds['longitude_bounds'] = ('longitude_bounds'),lonb
    ds['longitude_bounds'].attrs['units'] = 'Degrees E'

    ds['area'] = ('latitude','longitude'),area
    ds['area'].attrs['units'] = 'km^2'

    ds['ocean_mask'] = ('latitude','longitude'),mask
    ds['ocean_mask'].attrs['units'] = 'Ocean=1; Land=0'

    ds['Harvest'] = ('latitude','longitude'),Harvest
    ds['Harvest'].attrs['units'] = 'gDW/m^2'

    ds['Growth'] = ('latitude','longitude'),Growth
    ds['Growth'].attrs['units'] = 'gDW/m^2'

    ds['index_H'] = ('latitude','longitude'),index_H
    ds['index_H'].attrs['units'] = 'seaweed type'
    ds['index_H'].attrs['guide'] = '0=Eucheuma; 1=Sargassum; 2=Porphyra; 3=Saccharina; 4=Macrocystis'

    ds['nharv_H'] = ('latitude','longitude'),nharv_H
    ds['nharv_H'].attrs['units'] = 'number of harvests'

    encoding = {'area': {"dtype": np.float32, "zlib": True, "complevel": 4,},
                'Growth': {"dtype": np.float32, "zlib": True, "complevel": 4,},
                'Harvest': {"dtype": np.float32, "zlib": True, "complevel": 4,},
                'index_H': {"dtype": np.int32, "zlib": True, "complevel": 4, },
                'nharv_H': {"dtype": np.int32, "zlib": True, "complevel": 4, },
                'ocean_mask': {"dtype": "i1", "zlib": True, "complevel": 4}}
    ds.to_netcdf(out_nc_filepath,format='netCDF4',encoding=encoding)



def patch_SST_w_HYCOM(sst_nc,hycom_nc,out_nc):
    
    ds = xr.open_dataset(sst_nc,engine='netcdf4')
    ds_p = xr.open_dataset(hycom_nc,engine='netcdf4')

    ds.sst[...] = ds.sst.fillna(ds_p.t_mean.data)
   
    encoding = {"sst": {"dtype": np.float32, "zlib": True, "complevel": 4,
                        "chunksizes": (1, 360, 4320)}}
    ds.to_netcdf(out_nc, format='netCDF4', encoding=encoding)

    

if __name__ == "__main__":
    make_best_spp_nc(
        r"X:/data/mag/output/mc_paper_v9/mc_paper_v9_patch_Eucheuma_f0.nc",
        r'X:/data/mag/output/mc_paper_v9/Preferred_species_f0_mc_paper_v9_patch.nc',mc_results=True)
    make_best_spp_nc(
        r"X:/data/mag/output/mc_paper_v9/mc_paper_v9_patch_Eucheuma_f1.nc",
        r'X:/data/mag/output/mc_paper_v9/Preferred_species_f1_mc_paper_v9_patch.nc',mc_results=True)

    exit()
    make_best_spp_nc(r"/mnt/vat/data/mag/output/std/v9/std_lim_terms_v9_20230119_2017_Eucheuma_f0/mag0_output_std_lim_terms_v9_20230119_2017_Eucheuma_f0.nc",
                     r'/mnt/vat/data/mag/output/std/v9/Preferred_species_f0_std_lim_terms_v9_20230119.nc')
    make_best_spp_nc(r"/mnt/vat/data/mag/output/std/v9/std_lim_terms_v9_20230119_2017_Eucheuma_f1/mag0_output_std_lim_terms_v9_20230119_2017_Eucheuma_f1.nc",
                     r'/mnt/vat/data/mag/output/std/v9/Preferred_species_f1_std_lim_terms_v9_20230119.nc')

    exit()

    for y in range(2011,2020):
        patch_SST_w_HYCOM('/mnt/vat/data/CWM/SST/sst_%i_gridinterp.nc'%y,
                          r'/mnt/vat/data/CWM/hycom/HYCOM_ts_2017_9km.nc',
                          r'/mnt/vat/data/CWM/SST/sst_%i_gridinterp_patch.nc'%y)
    exit()



    patch_SST_w_HYCOM(r'/mnt/vat/data/CWM/SST/sst_2017_gridinterp.nc',
                      r'/mnt/vat/data/CWM/hycom/HYCOM_ts_2017_9km.nc',
                      r'/mnt/vat/data/CWM/SST/sst_2017_gridinterp_patched.nc')
    exit()

    #generate_hycom_9km_v2(r'X:\data\CWM\hycom',2017,method='nearest',one_day=False,ts=True)
    generate_hycom_9km_v2(r'/mnt/vat/data/CWM/hycom',2017,method='nearest',one_day=False,ts=True)
    exit()


    hycom_0_lon = list(range(140,152)) #(list(range(90,152)) + list(range(274,366)))[50:100]
    for i in hycom_0_lon:
        print('regridding: %i'%i)
        # regrid_hycom_to_m180(r'X:\data\CWM\hycom\hycom_ts_2017_z0_'+'%03i.nc'%i,
        #                  r'X:\data\CWM\hycom\regridded\hycom_ts_2017_z0_'+'%03i.nc'%i,
        #                  r'X:\data\CWM\hycom\hycom_ts_2017_z0_270.nc')
        regrid_hycom_to_m180(r'/mnt/vat/data/CWM/hycom/hycom_ts_2017_z0_' + '%03i.nc' % i,
                         r'/mnt/vat/data/CWM/hycom/regridded/hycom_ts_2017_z0_' + '%03i.nc' % i,
                         r'/mnt/vat/data/CWM/hycom/hycom_ts_2017_z0_271.nc')
    exit()
    
    for i in range(120,180):
        ds = xr.open_dataset(r'X:\data\CWM\hycom\hycom_ts_2017_z0_' +'%03i.nc'%i)
        print(i,ds['lon'].values[0])
        ds.close()
    exit()


    


    hycom_0_lon = (list(range(90,140)) + list(range(274,366)))[50:100]
    for i in hycom_0_lon:
        print('regridding: %i'%i)
        # regrid_hycom_to_m180(r'X:\data\CWM\hycom\hycom_ts_2017_z0_'+'%03i.nc'%i,
        #                  r'X:\data\CWM\hycom\regridded\hycom_ts_2017_z0_'+'%03i.nc'%i,
        #                  r'X:\data\CWM\hycom\hycom_ts_2017_z0_270.nc')
        regrid_hycom_to_m180(r'/mnt/vat/data/CWM/hycom/hycom_ts_2017_z0_' + '%03i.nc' % i,
                         r'/mnt/vat/data/CWM/hycom/regridded/hycom_ts_2017_z0_' + '%03i.nc' % i,
                         r'/mnt/vat/data/CWM/hycom/hycom_ts_2017_z0_271.nc')
    exit()







    interpolate_missing_cwm(r'X:\data\CWM\chla\chl_YYYY.nc', ['chl'], [2017,2018,2019], lat_filter=None, lat_interp=False,mask_fn=default_cwm_mask)
    exit()

    interpolate_missing_cwm(r'X:\data\CWM\chla\chl_YYYY.nc', ['chl'], np.arange(2003, 2020), lat_filter=None,
                            lat_interp=False, mask_fn=default_cwm_mask)

    exit()
    interpolate_missing_cwm_nc4(r'X:/data/CWM/ECMWF/ecmwf_cwm_YYYY_9km_1day.nc', ['swh_mean','mwp_mean'], [2017],
                            lat_filter=None,lat_interp=False,mask_fn=default_cwm_mask)

    interpolate_missing_cwm(r'X:\data\CWM\ECMWF\ecmwf_cwm_YYYY_9km.nc', ['swh_mean','mwp_mean'], np.arange(2003,2020),
                            lat_filter=None,lat_interp=False,mask_fn=default_cwm_mask)
    interpolate_missing_cwm(r'X:\data\CWM\chla\chl_YYYY.nc', ['chl'], np.arange(2003,2020), lat_filter=None, lat_interp=False,mask_fn=default_cwm_mask)

    exit()


    make_best_spp_nc(r"X:\data\mag\output\std\v9\std_lim_terms_v9_paper_CESMseed_2017_Eucheuma_f0\mag0_output_std_lim_terms_v9_paper_CESMseed_2017_Eucheuma_f0.nc",
                     r'X:\data\mag\output\std\v9\Preferred_species_f0_std_lim_terms_v9_paper_CESMseed.nc')
    make_best_spp_nc(r"X:\data\mag\output\std\v9\std_lim_terms_v9_paper_CESMseed_2017_Eucheuma_f1\mag0_output_std_lim_terms_v9_paper_CESMseed_2017_Eucheuma_f1.nc",
                     r'X:\data\mag\output\std\v9\Preferred_species_f1_std_lim_terms_v9_paper_CESMseed.nc')

    exit()

    make_best_spp_nc(r"X:\data\mag\output\std\v9\std_lim_terms_v9_paper_2017_Eucheuma_f0\mag0_output_std_lim_terms_v9_paper_2017_Eucheuma_f0.nc",
                     r'X:\data\mag\output\std\v9\Preferred_species_f0_std_lim_terms_v9_paper.nc')
    make_best_spp_nc(r"X:\data\mag\output\std\v9\std_lim_terms_v9_paper_2017_Eucheuma_f1\mag0_output_std_lim_terms_v9_paper_2017_Eucheuma_f1.nc",
                     r'X:\data\mag\output\std\v9\Preferred_species_f1_std_lim_terms_v9_paper.nc')
    exit()
    make_best_spp_nc(r"X:\data\mag\output\std\breakage_output\std_lim_terms_v8_paper_dQdtfix_gQ-0-1_Copernicus_2017_Eucheuma_f0\mag0_output_std_lim_terms_v8_paper_dQdtfix_gQ-0-1_Copernicus_2017_Eucheuma_f0.nc",
                     r'X:\data\mag\output\std\breakage_output\Preferred_species_f1_v8_paper_dQdtfix_gQ-0-1_Copernicus.nc')
    make_best_spp_nc(r"X:\data\mag\output\std\breakage_output\std_lim_terms_v8_paper_dQdtfix_gQ-0-1_EuchKs_2017_Eucheuma_f0\mag0_output_std_lim_terms_v8_paper_dQdtfix_gQ-0-1_EuchKs_2017_Eucheuma_f0.nc",
                     r'X:\data\mag\output\std\breakage_output\Preferred_species_f1_v8_paper_dQdtfix_gQ-0-1_newEuchKsVmax.nc')
    exit()
    make_best_spp_nc(r"X:\data\mag\output\std\breakage_output\std_lim_terms_v8_paper_2017_Eucheuma_f1\mag0_output_std_lim_terms_v8_paper_2017_Eucheuma_f1.nc",
                     r'X:\data\mag\output\std\breakage_output\Preferred_species_f1_v8_paper.nc')
    make_best_spp_nc(r"X:\data\mag\output\std\breakage_output\std_lim_terms_v8_paper_dQdtfix_gQ-0-1_2017_Eucheuma_f1\mag0_output_std_lim_terms_v8_paper_dQdtfix_gQ-0-1_2017_Eucheuma_f1.nc",
                     r'X:\data\mag\output\std\breakage_output\Preferred_species_f1_v8_paper_dQdtfix_gQ-0-1.nc')
    exit()



    EEZ_mask_USA(r'X:\data\CWM\regions_and_masks\World_EEZ_USA_only\eez_v11.shp',
              r'X:/data/mag/eez_masks.nc')

    exit()

    longitude, latitude, xlon, ylat = load_cwm_grid_no_mask()
    rg = netCDF4.Dataset(r'X:\data\mag\output\TEA_postprocessed\Preferred_species_f1_MC.nc',mode='a')
    rg.variables['lon'][...] = longitude
    rg.variables['lat'][...] = latitude
    rg.close()
    exit()


    make_best_spp_nc(r'X:\data\mag\output\std\lim_terms_v6\std_lim_terms_v6_f1native_5day_full_output_2017_Eucheuma_f1\mag0_output_std_lim_terms_v6_f1native_5day_full_output_2017_Eucheuma_f1.nc',
                     r'X:\data\mag\output\TEA_postprocessed\Preferred_species_f1_STD_only_Ben.nc')
    exit()

    # 0 = Euc; 1 = Sarg; 2 = Porph; 3= Sacc 4 = Macro
    plt.imshow(ds['Harvest'].values,cmap=plt.get_cmap("viridis"),vmin=0,vmax=5000)
    plt.figure()
    plt.imshow(ds['Harvest_q50'].values,cmap=plt.get_cmap("viridis"),vmin=0,vmax=5000)
    plt.title('q50')
    plt.show()
    exit()










    generate_ecco_darwin_v5_cwm(r'X:\data\seaweed2\ECCO_Darwin_v5\output\monthly\NO3',
                                r'X:\data\seaweed2\ECCO_Darwin_v5\llc_270\grid',
                                r'X:\data\seaweed2\ECCO_Darwin_v5\ECCO_Darwin_v5_cwm_2017_NO3_20m.nc')


    save_carib_gom_masks()
    exit()

    # generate_WOA_8day(r'X:\data\CWM\WOA\P', write_out=True, convert_from_per_kg=True, cubic=False,fast_i=False,depth=20,n_or_p='p')
    # generate_WOA_8day(r'X:\data\CWM\WOA\P', write_out=True, convert_from_per_kg=True, cubic=True,fast_i=False,depth=20,n_or_p='p')
    # generate_WOA_8day(r'X:\data\CWM\WOA\P', write_out=True, convert_from_per_kg=True, cubic=False,fast_i=False,depth=100,n_or_p='p')
    # generate_WOA_8day(r'X:\data\CWM\WOA\P', write_out=True, convert_from_per_kg=True, cubic=True,fast_i=False,depth=100,n_or_p='p')
    # generate_WOA_9km(r'X:\data\CWM\WOA\P\WOA18_P_8day_interp_cubic_20m.nc',r"X:\data\CWM\WOA\P\WOA18_P_8day_interp_9km_cubic_20m.nc",'cubic',in_var='P',out_var="PO4",out_units="mg PO4 m-3")
    # generate_WOA_9km(r'X:\data\CWM\WOA\P\WOA18_P_8day_interp_cubic_20m.nc',r"X:\data\CWM\WOA\P\WOA18_P_8day_interp_9km_nearest_20m.nc",'nearest',in_var='P',out_var="PO4",out_units="mg PO4 m-3")
    # generate_WOA_9km(r'X:\data\CWM\WOA\P\WOA18_P_8day_interp_cubic_100m.nc',r"X:\data\CWM\WOA\P\WOA18_P_8day_interp_9km_cubic_100m.nc",'cubic',in_var='P',out_var="PO4",out_units="mg PO4 m-3")
    # generate_WOA_9km(r'X:\data\CWM\WOA\P\WOA18_P_8day_interp_cubic_100m.nc',r"X:\data\CWM\WOA\P\WOA18_P_8day_interp_9km_nearest_100m.nc",'nearest',in_var='P',out_var="PO4",out_units="mg PO4 m-3")
    generate_WOA_9km_monthly(r'X:\data\CWM\WOA\P\WOA18_P_20m.nc',r"X:\data\CWM\WOA\P\WOA18_P_9km_cubic_20m.nc",'cubic',in_var='P',out_var="PO4",out_units="mg PO4 m-3")
    generate_WOA_9km_monthly(r'X:\data\CWM\WOA\P\WOA18_P_20m.nc',r"X:\data\CWM\WOA\P\WOA18_P_9km_nearest_20m.nc",'nearest',in_var='P',out_var="PO4",out_units="mg PO4 m-3")
    generate_WOA_9km_monthly(r'X:\data\CWM\WOA\P\WOA18_P_100m.nc',r"X:\data\CWM\WOA\P\WOA18_P_9km_cubic_100m.nc",'cubic',in_var='P',out_var="PO4",out_units="mg PO4 m-3")
    generate_WOA_9km_monthly(r'X:\data\CWM\WOA\P\WOA18_P_100m.nc',r"X:\data\CWM\WOA\P\WOA18_P_9km_nearest_100m.nc",'nearest',in_var='P',out_var="PO4",out_units="mg PO4 m-3")
    exit()


    # generate_WOA_8day(r'X:\data\CWM\WOA\N', write_out=True, convert_from_per_kg=True, cubic=False,fast_i=False,depth=20)
    # generate_WOA_8day(r'X:\data\CWM\WOA\N', write_out=True, convert_from_per_kg=True, cubic=True,fast_i=False,depth=20)
    # generate_WOA_8day(r'X:\data\CWM\WOA\N', write_out=True, convert_from_per_kg=True, cubic=False,fast_i=False,depth=100)
    # generate_WOA_8day(r'X:\data\CWM\WOA\N', write_out=True, convert_from_per_kg=True, cubic=True,fast_i=False,depth=100)
    # generate_WOA_9km(r'X:\data\CWM\WOA\N\WOA18_N_8day_interp_cubic_20m.nc',r"X:\data\CWM\WOA\N\WOA18_N_8day_interp_9km_cubic_20m.nc",'cubic')
    # generate_WOA_9km(r'X:\data\CWM\WOA\N\WOA18_N_8day_interp_cubic_20m.nc',r"X:\data\CWM\WOA\N\WOA18_N_8day_interp_9km_nearest_20m.nc",'nearest')
    # generate_WOA_9km(r'X:\data\CWM\WOA\N\WOA18_N_8day_interp_cubic_100m.nc',r"X:\data\CWM\WOA\N\WOA18_N_8day_interp_9km_cubic_100m.nc",'cubic')
    # generate_WOA_9km(r'X:\data\CWM\WOA\N\WOA18_N_8day_interp_cubic_100m.nc',r"X:\data\CWM\WOA\N\WOA18_N_8day_interp_9km_nearest_100m.nc",'nearest')
    generate_WOA_9km_monthly(r'X:\data\CWM\WOA\N\WOA18_N_20m.nc',r"X:\data\CWM\WOA\N\WOA18_N_9km_cubic_20m.nc",'cubic')
    generate_WOA_9km_monthly(r'X:\data\CWM\WOA\N\WOA18_N_20m.nc',r"X:\data\CWM\WOA\N\WOA18_N_9km_nearest_20m.nc",'nearest')
    generate_WOA_9km_monthly(r'X:\data\CWM\WOA\N\WOA18_N_100m.nc',r"X:\data\CWM\WOA\N\WOA18_N_9km_cubic_100m.nc",'cubic')
    generate_WOA_9km_monthly(r'X:\data\CWM\WOA\N\WOA18_N_100m.nc',r"X:\data\CWM\WOA\N\WOA18_N_9km_nearest_100m.nc",'nearest')
    exit()

    #combine_WOA(r'X:\data\CWM\WOA\N', convert_from_per_kg=True,depth_mean=20.,var='n')
    #combine_WOA(r'X:\data\CWM\WOA\N', convert_from_per_kg=True,depth_mean=100.,var='n')
    combine_WOA(r'X:\data\CWM\WOA\P', convert_from_per_kg=True,depth_mean=20.,var='p')
    combine_WOA(r'X:\data\CWM\WOA\P', convert_from_per_kg=True,depth_mean=100.,var='p')
    exit()


    interpolate_missing_cwm(r'X:\data\CWM\PAR\par_YYYY.nc', ['par'], np.arange(2003,2020), lat_interp=False,mask_fn=default_cwm_mask)
    interpolate_missing_cwm(r'X:\data\CWM\SST\sst_YYYY.nc', ['sst'], np.arange(2003,2020), lat_filter=None,lat_interp=False,mask_fn=default_cwm_mask)
    exit()

    save_new_analysis_masks(default_cwm_mask,r'X:\data\CWM\regions_and_masks\analysis_masks_20220412.nc')
    exit()


    combine_WOA(r'X:\data\CWM\WOA\P', convert_from_per_kg=True, var='p')
    exit()



    exit()

    make_mask_from_hycom(r'X:\data\CWM\hycom\hycom_2016_z0_121.nc', r'X:\data\CWM\regions_and_masks\cwm_mask_20220412_from_hycom.txt')
    exit()

    generate_ecmwf_9km(r'X:\data\CWM\ECMWF', 2017, ec_vars=["swh", "mwp"], method="nearest", one_day=True)    
    exit()

    generate_hycom_9km_v2(r'X:\data\CWM\hycom',2017,method='nearest',one_day=True)
    exit()    

    MPA_mask([r'/mnt/reservior/data/CWM/regions_and_masks/WDPA/shp0/WDPA_Dec2021_Public_shp-polygons.shp',
              r'/mnt/reservior/data/CWM/regions_and_masks/WDPA/shp1/WDPA_Dec2021_Public_shp-polygons.shp',
              r'/mnt/reservior/data/CWM/regions_and_masks/WDPA/shp2/WDPA_Dec2021_Public_shp-polygons.shp'], 
             r'/mnt/reservior/data/CWM/regions_and_masks/WDPA/WDPA_CWM.nc', 
             minimum_sq_km=20.0)
    exit()    
    
    MPA_mask([r'/mnt/reservior/data/CWM/regions_and_masks/WDPA/CA_NMS_only.shp'], 
             r'/mnt/reservior/data/CWM/regions_and_masks/WDPA/CA_NMS_only.c', 
             minimum_sq_km=20.0)
    exit()   




    mode_filter_nc_append(r'/mnt/reservior/data/mag/output/seed_paper_v5/Saccharina_s1_t0_freq180_sp120_nh2_frac0.8_kg1.35_kcap2000_f0_seed_month_multiyear.nc','seed_month',ksize=9)
    exit()
    
    for i in range(32,92):
        print('regridding: %i'%i)
        regrid_hycom_to_m180(r'X:\data\CWM\hycom\hycom_2017_z0_'+'%03i.nc'%i,
                         r'X:\data\CWM\hycom\regridded\hycom_2017_z0_'+'%03i.nc'%i,
                         r'X:\data\CWM\hycom\hycom_2017_z0_270.nc')
    exit()

    for i in range(25,180):
        ds = xr.open_dataset(r'X:\data\CWM\hycom\hycom_2017_z0_' +'%03i.nc'%i)
        print(i,ds['lon'].values[0])
        ds.close()
    exit()

    test_plot_nc(r'X:\data\CWM\hycom\HYCOM_2017_9km.nc','speed_mean')
    exit()

    
    generate_hycom_9km_v2(r'X:\data\CWM\hycom',2017,method='nearest')
    exit()



    
    test_plot_nc(r'X:\data\CWM\hycom\HYCOM_2016_9km.nc','speed_mean')
    test_plot_nc(r'X:\data\CWM\SST\sst_2016.nc', 'sst')
    test_plot_nc(r'X:\data\CWM\PAR\par_2016.nc', 'par')
    test_plot_nc(r'X:\data\CWM\chla\chl_2016.nc', 'chl')    
    test_plot_nc(r'X:\data\CWM\ECMWF\ecmwf_cwm_2016_9km.nc', 'swh_mean')
    test_plot_nc(r'X:\data\CWM\ECMWF\ecmwf_cwm_2016_9km.nc', 'swh_max')
    test_plot_nc(r'X:\data\CWM\ECMWF\ecmwf_cwm_2016_9km.nc', 'mwp_mean')
    test_plot_nc(r'X:\data\CWM\ECMWF\ecmwf_cwm_2016_9km.nc', 'mwp_max')
    exit()
    
    mode_filter_nc_append(r'/mnt/reservior/data/mag/output/seed_paper_v5/Macrocystis_s1_t1_freq220_sp150_nh2_frac0.8_kg1.35_kcap2000_f0_seed_month_multiyear.nc','seed_month',ksize=9)
    mode_filter_nc_append(r'/mnt/reservior/data/mag/output/seed_paper_v5/Eucheuma_s2_t1_freq364_sp320_nh8_frac0.8_kg0.80_kcap3000_f0_seed_month_multiyear.nc','seed_month',ksize=9)
    mode_filter_nc_append(r'/mnt/reservior/data/mag/output/seed_paper_v5/Sargassum_s2_t1_freq364_sp320_nh8_frac0.8_kg0.40_kcap800_f0_seed_month_multiyear.nc','seed_month',ksize=9)
    mode_filter_nc_append(r'/mnt/reservior/data/mag/output/seed_paper_v5/Porphyra_s1_t1_freq150_sp110_nh6_frac0.8_kg0.08_kcap200_f0_seed_month_multiyear.nc','seed_month',ksize=9)
    exit()

    mode_filter_nc_append(r'/mnt/reservior/data/mag/output/seed_paper_v5/Saccharina_s1_t0_freq180_sp120_nh2_frac0.8_kg1.35_kcap2000_f0_seed_month_multiyear.nc','seed_month',ksize=9)
    exit()    



    generate_mask_from_SST()
    exit()

    test_plot_nc(r"/mnt/reservior/data/mag/output/std/multiyear_v4/2019_std_multiyear_Saccharina_f0/mag0_output_2019_std_multiyear_Saccharina_f0.nc", "Growth2")
    exit()    

    interpolate_chl_nc(r"/mnt/reservior/data/CWM/chla/chl_2019.nc",2019)
    exit()    


    test_plot_nc(r"/mnt/reservior/data/CWM/chla/chl_2019.nc", "chl")
    exit()

    generate_fseq_10(
        r"X:/data/CWM/fseq/fseq_cwm_",
        r"X:/data/CWM/fseq/fseq_cwm_by_100y_linear_interp.nc",
        method="linear",
    )

    # MEOW_masks(r'X:/data/CWM/regions_and_masks/MEOW/meow_ecos.shp',
    #          r'X:/data/mag/MEOW_masks.nc')

    # LME66_masks(r'X:/data/CWM/regions_and_masks/LME66/LMEs66.shp',
    #          r'X:/data/mag/LME66_masks.nc')

    # FAO_masks(r'X:/data/CWM/regions_and_masks/FAO_AREAS_NOCOASTLINE/FAO_AREAS_NOCOASTLINE.shp',
    #          r'X:/data/mag/FAO_masks.nc')
    # generate_ship_traffic(r'X:/data/CWM/regions_and_masks/Global Ship Density/shipdensity_global.tif')
    # generate_fseq(r'X:/data/CWM/fseq_cwm.mat',r'X:/data/CWM/fseq_cwm.nc',method='nearest')
    # numba.set_num_threads(20)
    # EEZ_masks(r'/mnt/reservior/data/CWM/regions_and_masks/World_EEZ_v11_20191118/eez_v11.shp',
    #          r'/mnt/reservior/data/mag/eez_masks.nc')
