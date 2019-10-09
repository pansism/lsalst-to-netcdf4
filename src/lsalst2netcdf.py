# -*- coding: utf-8 -*-

"""
This script gets as input one or more MSG-SEVIRI LST HDF5 products (MLST LSA-001)
from EUMETSAT's Satellite Application Facility on Land Surface Analysis (LSA SAF)
and extracts---for the given bounding box and for each input file---the LST, QF
and errorbar datasets. The extracted datasets are first resampled to a regular
lat-lon grid and then stacked according to their acquisition time. Finally, the
resulting LST, QF and errorbar stacks are added to a netCDF4 file that can be
readily opened by GDAL.

MSG-SEVIRI is a geostationary meteorological satellite operated by EUMETSAT that
acquires image data of Europe, Africa and South America every 15 min in 12 spectral
bands.

To download MLST LSA-001 data go to:
    https://landsaf.ipma.pt/en/products/land-surface-temperature/lst/

USAGE:   To use this script call the function LSALSTstack2NetCDF().

CAUTION: This script assumes that the input directory contains ONLY
         unzipped MLST LSA-001 hdf5 files.


-------------------------------------------------------------------------------------
Current version: 1.0 | Supports Python 3.7

Changelog: 
    25/09/2019: v1.0 - Release Date.          
-------------------------------------------------------------------------------------

AUTHOR:       Panagiotis Sismanidis
AFFILIATION:  National Observatory of Athens [www.astro.noa.gr/en/main/]

This software is provided under the MIT license.

enjoy!

"""

import concurrent.futures 
import h5py
import netCDF4 as nc
import numpy as np
import os
from datetime import datetime
from math import sin, cos, tan, asin, atan, sqrt, pow, radians, degrees


def _GetProductAttributes(h5_file):
    """Returns a dictionary with the HDF's global attributes and dataset names.
       
    Arguments:
        h5_file {string} -- A valid HDF5 file.

    Returns:
        attrs_dic {dic} -- The HDF5's global attributes and dataset names.
    """
    attrs_dic = {}
    with h5py.File(h5_file, mode="r") as h5:
        for key in h5.attrs.keys():
            attrs_dic[key] = h5.attrs[key]  
            attrs_dic["DSNAMES"] = [ds for ds in h5.keys()]
        
    return attrs_dic


def latlon2rowcol(lat, lon, product_attrs):
    """This function solves the forward GEOS projection function for\
       the given lat, lon and returns the corresponding row, column.

       SOURCE: Wolf, R.; Just, D. LRIT/HRIT global specification.\
       Coordination Group for Meteorological Satellites, 1999, 2.6.
       Url: https://www.cgms-info.org/documents/pdf_cgms_03.pdf

    Arguments:
        lat {float} -- The latitude of a point on the Earth's surface (in decimal degrees).
        lon {float} -- The longitude of a point on the Earth's surface (in decimal degrees).
        product_attrs {dict} -- The MLST LSA-001 global attributes.
    
    Returns:
        col {int16} -- The HDF5 column that corresponds to the given Lat/Lon.
        row {int16} -- The HDF5 row that corresponds to the given Lat/Lon.
    """
    H = 42164.0
    R_polar = 6356.5838
    R_equator = 6378.1690
    e = sqrt(1.0 - pow(R_polar / R_equator, 2))

    lat = radians(lat)
    lon = radians(lon)
    sub_lon = radians(product_attrs["SUB_SATELLITE_POINT_START_LON"])
    c_lat = atan((pow(R_polar, 2) / pow(R_equator, 2)) * tan(lat))

    r_l = R_polar / sqrt(1 - pow(e, 2) * pow(cos(c_lat), 2))

    r_x = -1.0 * r_l * cos(c_lat) * sin(lon - sub_lon)
    r_y = r_l * sin(c_lat)
    r_z = H - r_l * cos(c_lat) * cos(lon - sub_lon)
    r_n = sqrt(pow(r_x, 2) + pow(r_y, 2) + pow(r_z, 2))

    x = degrees(atan(-1.0 * r_x / r_z))
    y = degrees(asin(-1.0 * r_y / r_n))

    # Subtract 1 from COFF and LOFF to change origin from (1,1) to (0,0).
    col = int(product_attrs["COFF"]-1 + round(x * pow(2.0, -16) * product_attrs["CFAC"]))
    row = int(product_attrs["LOFF"]-1 + round(y * pow(2.0, -16) * product_attrs["LFAC"]))
    if col > product_attrs["NC"] or row > product_attrs["NL"]:
        raise ValueError(f"A col/row value cannot be greater than {product_attrs['NC']}.")

    return col, row


def _BBox2RowColGrids(bbox, hres, vres, product_attrs):
    """This function returns a regular grid that covers the given BBox and\
       stores in each cell the coresponding HDF5 row and column values.
    
    Arguments:
        bbox {dic} -- The bounding box coordinates (in decimal degrees)
        hres {float} -- The horizontal grid resolution (in decimal degrees)
        vres {float} -- The vertical grid resolution (in decimal degrees)
        product_attrs {dic} -- The MLST LSA-001 global attributes.
    
    Returns:
        row_grid {np.array} -- A regular grid with the HDF5 row of each grid cell.
        col_grid {np.array} -- A regular grid with the HDF5 column of each grid cell.
    """
    if hres > 0 and vres > 0:

        grid_dim = (
            abs(bbox["north_lat"] - bbox["south_lat"]) / vres + 1,
            abs(bbox["west_lon"] - bbox["east_lon"]) / hres + 1,
        )
        
        lats = np.linspace(
            start=bbox["north_lat"], stop=bbox["south_lat"], num=grid_dim[0], endpoint=True
        )
        
        lons = np.linspace(
            start=bbox["west_lon"], stop=bbox["east_lon"], num=grid_dim[1], endpoint=True
        )

        row_grid = np.full(shape=(len(lats), len(lons)), fill_value=-1, dtype='int16')
        col_grid = np.full(shape=(len(lats), len(lons)), fill_value=-1, dtype='int16')

        for r, lat in enumerate(lats, start=0):
            for c, lon in enumerate(lons, start=0):   
                col, row = latlon2rowcol(lat, lon, product_attrs)
                row_grid[r, c] = row
                col_grid[r, c] = col
                
        return row_grid, col_grid
    else:
        raise ValueError("The grid resolution cannot be negative or zero.")
    

def _ReadH5DatasetAsArray(h5_file, row_grid, col_grid, dataset):    
    """This function reads data from a valid MLST LSA-001 dataset; it then resamples them\
       to a regular lat-lon grid; and calulates a flag that indicates if all the data\
       inside the BBox are missing due to clouds.

    Arguments:
        h5_file {HDF5 file object} -- A valid MLST LSA-001 HDF5 file object.
        row_grid {np.array} -- A regular grid with the HDF5 row of each grid cell.
        col_grid {np.array} -- A regular grid with the HDF5 column of each grid cell.
        dataset {string} -- The dataset name.

    Returns:
        data_array {np.array} -- An array with the hdf5 data that fall inside the BBox\
                                 resampled to a regular lat-ln grid.
        validscene_flag {boolean} -- A flag that is False if all the data inside the BBox\
                                     are missing due to clouds.
    """
    ds = h5_file[dataset]
    h5_data = ds[:]
    data_array = np.full(
        shape=row_grid.shape, fill_value=ds.attrs["MISSING_VALUE"], dtype=ds.dtype
    )
    for r in range(row_grid.shape[0]):
        for c in range(row_grid.shape[1]):
            ds_row = row_grid[r, c] 
            ds_col = col_grid[r, c]
            data_array[r, c] = h5_data[ds_row, ds_col]

    if np.mean(data_array) != ds.attrs["MISSING_VALUE"]:
        validscene_flag = True
        return data_array, validscene_flag
    else:
        validscene_flag = False
        return None, validscene_flag
    

def _GetDataFromHDF5(h5_filepath, row_grid, col_grid, product_attrs):
    """This function opens a valid MLST LSA-001 HDF5 file and gets: a flag\
       indicating if the data inside the BBox are valid; the corresponding \
       gridded LST, QF and errorbars datasets; and theLST acquisition time.

    Arguments:
        h5_filepath {string} -- The absolute filepath of a valid MLST LSA-001 HDF5 file.
        row_grid {np.array} -- A regular grid with the HDF5 row of each grid cell.
        col_grid {np.arrray} -- A regular grid with the HDF5 column of each grid cell.
        product_attrs {dic} -- The global attributes of a LSALST file.

    Returns:
        [list] -- A list with (1) a flag indicating if the data inside the BBox\
                  are valid; (2,3,4) the gridded LST, QF and errorbars datasets;\
                  and (5) the LST acquisition time.
    """
    with h5py.File(h5_filepath, mode="r") as h5_file:
        if h5_file.attrs["PRODUCT_TYPE"] == product_attrs["PRODUCT_TYPE"]:
            LST_array, validscene_flag = _ReadH5DatasetAsArray(
                h5_file, row_grid, col_grid, product_attrs["DSNAMES"][0]
            )
            if validscene_flag == True:
                QF_array, _ = _ReadH5DatasetAsArray(
                    h5_file, row_grid, col_grid, product_attrs["DSNAMES"][1]
                )
                errorbar_array, _ = _ReadH5DatasetAsArray(
                    h5_file, row_grid, col_grid, product_attrs["DSNAMES"][2]
                )
                time_string = h5_file.attrs["IMAGE_ACQUISITION_TIME"].decode('UTF-8')
                sensing_time = datetime.strptime(time_string, '%Y%m%d%H%M%S')

                out_list = [validscene_flag, LST_array, QF_array, errorbar_array, sensing_time]
            else:
                out_list =  [validscene_flag, None]

    return out_list



def LSALSTstack2NetCDF(h5dir, savedir, savename, latN, latS, lonW, lonE, hres=0.05, vres=0.05):
    """This function extracts the LST, QF and errorbar datasets that fall inside the given BBox from one\
       or more MLST LSA-001 HDF5 files. The extracted datasets are first resampled to a regular lat-lon grid,\
       and then stacked according to their acquistion time. The resulting LST, QF and errorbar  stacks are\
       then stored in a NetCDF4 file with all the appropriate metadata.

       WORKFLOW:
        1) Get a list wih the input MLST-LSA001 HDF5 files.
        2) For the given BBox make a grid with the corresponding HDF5 row, columns.
        3) Open a writable netCDF4 file and prepare its attributes.
        4) Start reading the input HDF5 files launching multiple processes and\
           for each file extract the LST, QF and errorbar dataset in a regular grid\
           using the corresponding row, columns of step 3. Add the data to the open\
           netCDF4 file by stacking them according to their acquistion time.

    Arguments:
        h5dir {string} -- The absolute path of a folder that includes ONLY unzipped MLST LSA001 hdf5 files. 
        savedir {string} -- The absolute path of the folder where the output NetCDF4 will be stored.
        savename {string} -- The name of the output NetCDF4 file (without the extension).
        latN {float} -- The BBox's northernmost latitude.
        latS {float} -- The BBox's southernmost latitude.
        lonW {float} -- The BBox's westernmost longitude.
        lonE {float} -- The BBox's easternmost longitude.

    Keyword Arguments:
        hres {float} -- The horizontal grid resolution (default: {0.05})
        vres {float} -- The vertical grid resolution (default: {0.05})
    """
    start = datetime.now()

    try:
        h5_files = [os.path.join(h5dir, f) for f in os.listdir(h5dir)]
        prod_attrs = _GetProductAttributes(h5_files[0]) 
    except FileNotFoundError as err: print(err)

    if latN > latS and lonE > lonW:
        bbox = {"north_lat": latN, "south_lat": latS, "west_lon": lonW, "east_lon": lonE}
    else: raise ValueError("latN and lonE must by greater than latS and lonW, respectively.")

    row_grid, col_grid = _BBox2RowColGrids(bbox, vres, hres, prod_attrs)
    
    print(f"\n{LSALSTstack2NetCDF.__name__} started at {start.strftime('%I:%M%p')}.")
    print(f"\nFiles found in input folder: {len(h5_files)}\n")
    print("Input Bounding Box & Grid Resolution")
    print("="*36)
    print(f"{'Northernmost Latitide:':<26} {bbox['north_lat']:0.2f} deg")
    print(f"{'Southernmost Latitide:':<26} {bbox['south_lat']:0.2f} deg")
    print(f"{'Westernmost Longitude:':<26} {bbox['west_lon']:0.2f} deg")
    print(f"{'Easternmost Longitude:':<26} {bbox['east_lon']:0.2f} deg")
    print(f"{'Grid Resolution for Lats:':<26} {vres} deg")
    print(f"{'Grid Resolution for Lons:':<26} {hres} deg")

    # Open a netCDF4 file and start adding the MLST data.
    if not os.path.exists(savedir): os.makedirs(savedir)
    savepath = os.path.join(savedir, savename + ".nc")
    
    with nc.Dataset(savepath, mode="w",  format="NETCDF4_CLASSIC", WRITE_GDAL_TAGS="YES") as ncfile:
        
        ncfile.title = "Stack of Gridded SEVIRI LST from LandSAF"
        ncfile.date_created = f"Created: " + datetime.now().strftime("%Y-%m-%d %H:%M")
        ncfile.createDimension("time", None)
        ncfile.createDimension("lat", row_grid.shape[0])
        ncfile.createDimension("lon", row_grid.shape[1])

        # netcdf uses as origin the cell center coordinates and not the upper-left or lower-right.
        # To address this, the lat, lon arraus are adjusted by adding and subtacting half a pixel.
        lat = ncfile.createVariable('lat', np.float32, ('lat',))
        lat.units = 'degrees_north'
        lat[:] = np.linspace(
            bbox["north_lat"] - hres / 2, bbox["south_lat"] + hres / 2, num=row_grid.shape[0], endpoint=True
        ) 
        lon = ncfile.createVariable('lon', np.float32, ('lon',))
        lon.units = 'degrees_east'   
        lon[:] = np.linspace(
            bbox["west_lon"] + hres / 2, bbox["east_lon"] - hres / 2, num=row_grid.shape[1], endpoint=True
        )

        crs = ncfile.createVariable("WGS84", "c")
        crs.spatial_ref = """GEOGCS["WGS 84",]\
        DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],\
        AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],\
        UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]"""

        time = ncfile.createVariable('datetimes', np.float32, ('time',))
        time.units = 'hours since 0001-01-01 00:00:00'
        time.calendar = 'gregorian'

        LST = ncfile.createVariable('LST', np.int16, ('time','lat','lon'), fill_value=-8000)
        LST.long_name = "Land Surface Temperature"
        LST.units = "Celsius Degrees"
        LST.grid_mapping = "WGS84"
        LST.scale_factor = 100.0
        LST.add_offset = 0.0
            
        QF = ncfile.createVariable('qflags', np.int16, ('time','lat','lon'), fill_value=-9999)
        QF.long_name = "LandSAF LST quality flags"
        QF.grid_mapping = "WGS84"
        QF.scale_factor = 1.0
        QF.add_offset = 0.0
        
        errorbars = ncfile.createVariable('errorbars', np.int16, ('time','lat','lon'), fill_value=-1)
        errorbars.long_name = "LST error bars"
        errorbars.units = "Celsius Degrees"
        errorbars.grid_mapping = "WGS84"
        errorbars.scale_factor = 100.0
        errorbars.add_offset = 0.0
        
        # Launch parallel tasks to process the input HDF5 files.
        args_list = [(h5, row_grid, col_grid, prod_attrs) for h5 in h5_files]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            h5_data = executor.map(_GetDataFromHDF5, *zip(*args_list))
            i=0
            for data in h5_data:
                if data[0] == True:
                    LST[i,:,:] = data[1]
                    QF[i,:,:] = data[2]
                    errorbars[i,:,:] = data[3]
                    time[i] = nc.date2num(data[4], units=time.units, calendar=time.calendar)
                    i+=1

        nc_dims = LST.shape

    elapsed_time = datetime.now()-start
    print("\n" + "-"*60)
    print(f"Processing completed in {elapsed_time.total_seconds():.01f} sec.") #
    print(f"NetCDF4 data dimensions: {nc_dims}")
    print(f"HDF5 files discarded due to cloud gaps: {len(h5_files)-nc_dims[0]}")
    print(f"Output netCDF4 saved in: {savepath}")

if __name__ == "__main__":

    project_folder = "/Users/panosis/Dropbox/MyCodeRepository/sandbox/Data"
    datadir = "/Users/panosis/Dropbox/MyCodeRepository/sandbox/Data/lsalst_h5_sample"

    LSALSTstack2NetCDF(savedir=project_folder, savename="downscalingtest_8", h5dir=datadir, latN=38.5, latS=37.5, lonW=23.2, lonE=24.2)
