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

USAGE:  python lsalst2netcdf.py h5dir savedir savename latN, latS, lonW, lonE -r 0.05 0.05

CAUTION: This script assumes that the input directory contains ONLY
         unzipped MLST LSA-001 hdf5 files.


-------------------------------------------------------------------------------------
Current version: 1.2 | Supports Python 3.7

Changelog: 
    25/09/2019: v1.0 - Release Date. 
    09/10/2019: v1.1 - Fixed a bug in geocoding.  
    22/01/2020: v1.2 - Added a CLI.         
-------------------------------------------------------------------------------------

AUTHOR:       Panagiotis Sismanidis
AFFILIATION:  National Observatory of Athens [www.astro.noa.gr/en/main/]

This software is provided under the MIT license.

enjoy!

"""

import argparse
import concurrent.futures
import h5py
import netCDF4 as nc
import numpy as np
import os, sys
from datetime import datetime
from math import sin, cos, tan, asin, atan, sqrt, pow, radians, degrees
from devtools import debug


def __progressbar(total, iteration, message):
    """
    Displays or updates a console progress bar.
    Original Source: https://stackoverflow.com/a/45868571/11655162
    """
    barLength, status = 30, ""
    progress = float(iteration) / float(total)
    if progress >= 1.0:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r{} [{}] {:.0f}% [Working on h5 {} of {}] {}".format(
        message,
        "#" * block + "-" * (barLength - block),
        round(progress * 100, 0),
        iteration,
        total,
        status,
    )
    sys.stdout.write(text)
    sys.stdout.flush()

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
    col = int(
        product_attrs["COFF"] - 1 + round(x * pow(2.0, -16) * product_attrs["CFAC"])
    )
    row = int(
        product_attrs["LOFF"] - 1 + round(y * pow(2.0, -16) * product_attrs["LFAC"])
    )
    if col > product_attrs["NC"] or row > product_attrs["NL"]:
        raise ValueError(
            f"A col/row value cannot be greater than {product_attrs['NC']}."
        )

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
            abs(bbox["north_lat"] - bbox["south_lat"]) // vres + 1,
            abs(bbox["west_lon"] - bbox["east_lon"]) // hres + 1,
        )

        print("\nbbox:", bbox)
        print("grid_dim:", grid_dim[0])
        # debug(locals())
        lats = np.linspace(
                start=bbox["north_lat"],
                stop=bbox["south_lat"],
                num=int(grid_dim[0]),
                endpoint=True,
                )

        lons = np.linspace(
            start=bbox["west_lon"],
            stop=bbox["east_lon"],
            num=int(grid_dim[1]),
            endpoint=True,
        )

        row_grid = np.full(shape=(len(lats), len(lons)), fill_value=-1, dtype="int16")
        col_grid = np.full(shape=(len(lats), len(lons)), fill_value=-1, dtype="int16")

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
        shape=row_grid.shape, fill_value=ds.attrs["MISSING_VALUE"], dtype=np.float32
    )
    for r in range(row_grid.shape[0]):
        for c in range(row_grid.shape[1]):
            ds_row = row_grid[r, c]
            ds_col = col_grid[r, c]
            data_array[r, c] = h5_data[ds_row, ds_col] / ds.attrs["SCALING_FACTOR"]

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
                time_string = h5_file.attrs["IMAGE_ACQUISITION_TIME"].decode("UTF-8")
                sensing_time = datetime.strptime(time_string, "%Y%m%d%H%M%S")

                out_list = [
                    validscene_flag,
                    LST_array,
                    QF_array,
                    errorbar_array,
                    sensing_time,
                ]
            else:
                out_list = [validscene_flag, None]

    return out_list

def LSALSTstack2NetCDF(
    h5dir, savedir, savename, latN, latS, lonW, lonE, hres=0.05, vres=0.05
):
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
    except FileNotFoundError as err:
        print(err)

    # Estimate the centre coordinates of the outermost pixels using the input
    # BBox coordinates (they refer to the Upper-left / Lower-Right corners
    # of the outermost pixels).
    if latN > latS and lonE > lonW:

        bbox = {
            "north_lat": latN - vres / 2,
            "south_lat": latS + vres / 2,
            "west_lon": lonW + hres / 2,
            "east_lon": lonE - hres / 2,
        }
    else:
        raise ValueError(
            "latN and lonE must by greater than latS and lonW, respectively."
        )

    row_grid, col_grid = _BBox2RowColGrids(bbox, vres, hres, prod_attrs)

    print(f"\n{LSALSTstack2NetCDF.__name__} started at {start.strftime('%I:%M%p')}.")
    print(f"\nFiles found in input folder: {len(h5_files)}\n")
    print("Input Bounding Box & Grid Resolution")
    print("=" * 36)
    print(f"{'Northernmost Latitide:':<26} {latN:0.3f} deg")
    print(f"{'Southernmost Latitide:':<26} {latS:0.3f} deg")
    print(f"{'Westernmost Longitude:':<26} {lonW:0.3f} deg")
    print(f"{'Easternmost Longitude:':<26} {lonE:0.3f} deg")
    print(f"{'Grid Resolution for Lats:':<26} {vres} deg")
    print(f"{'Grid Resolution for Lons:':<26} {hres} deg")

    # Open a netCDF4 file and start adding the MLST data.
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = os.path.join(savedir, savename + ".nc")

    with nc.Dataset(savepath, "w", format="NETCDF4_CLASSIC", scale="False") as ncfile:

        ncfile.title = "Gridded SEVIRI LST"
        ncfile.description = """Stack of multi-temporal SEVIRI LST downloaded from LandSAF. Each 
        dataset has been clipped to the given lat-lon extend and resampled to a regular grid. The 
        corresponding quality flags and uncertainty estimates (errorbars) are also provided."""
        ncfile.keywords = "LST, Land Surface Temperature, SEVIRI, LandSAF"
        ncfile.instrument = "MSG-SEVIRI"
        ncfile.acknowledgement = (
            "The original source of the MLST LSA-001 data is LandSAF (landsaf.ipma.pt)"
        )
        ncfile.date_created = f"Created: " + datetime.now().strftime("%Y-%m-%d %H:%M")

        ncfile.createDimension("time", None)
        ncfile.createDimension("lat", row_grid.shape[0])
        ncfile.createDimension("lon", row_grid.shape[1])

        lat = ncfile.createVariable("lat", np.float32, ("lat",))
        lat.long_name = "Latitude"
        lat.units = "degrees_north"
        lat[:] = np.linspace(
            bbox["north_lat"], bbox["south_lat"], num=row_grid.shape[0], endpoint=True
        )
        lon = ncfile.createVariable("lon", np.float32, ("lon",))
        lon.long_name = "Longitude"
        lon.units = "degrees_east"
        lon[:] = np.linspace(
            bbox["west_lon"], bbox["east_lon"], num=row_grid.shape[1], endpoint=True
        )

        crs = ncfile.createVariable("crc", np.int8)
        crs.horizontal_datum_name = "WGS84"
        crs.grid_mapping_name = "latitude_longitude"
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.epsg_code = "4326"
        crs.ogc_wkt = """GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]"""

        time = ncfile.createVariable("datetimes", np.float32, ("time",))
        time.units = "hours since 2000-01-01 00:00:00"
        time.calendar = "standard"

        LST = ncfile.createVariable(
            "LST", np.float32, ("time", "lat", "lon"), fill_value=-80
        )
        LST.long_name = "Land Surface Temperature"
        LST.units = "Celsius Degrees"
        LST.valid_range = (-80.0, 70.0)
        LST.scale_factor = 1.0
        LST.add_offset = 0.0
        LST.grid_mapping = "crc"

        QF = ncfile.createVariable(
            "qflags", np.int16, ("time", "lat", "lon"), fill_value=0
        )
        QF.long_name = "LSALST quality flags"
        QF.units = "Dimensionless"
        QF.valid_range = (0, 14238)
        QF.scale_factor = 1.0
        QF.add_offset = 0.0
        QF.grid_mapping = "crc"

        errorbars = ncfile.createVariable(
            "errorbars", np.float32, ("time", "lat", "lon"), fill_value=-0.01
        )
        errorbars.long_name = "LST uncertainty estimates"
        errorbars.units = "Celsius Degrees"
        errorbars.valid_range = (0, np.PINF)
        errorbars.scale_factor = 1.0
        errorbars.add_offset = 0.0
        errorbars.grid_mapping = "crc"

        # Launch parallel tasks to process the input HDF5 files.
        args_list = [(h5, row_grid, col_grid, prod_attrs) for h5 in h5_files]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            h5_data = executor.map(_GetDataFromHDF5, *zip(*args_list))
            i = 0
            for data in h5_data:

                __progressbar(len(h5_files), i + 1, message="Progress:")

                if data[0] == True:
                    LST[i, :, :] = data[1]
                    QF[i, :, :] = data[2]
                    errorbars[i, :, :] = data[3]
                    time[i] = nc.date2num(
                        data[4], units=time.units, calendar=time.calendar
                    )
                    i += 1

        nc_dims = LST.shape

    elapsed_time = datetime.now() - start
    print("\n" + "-" * 60)
    print(f"Processing completed in {elapsed_time.total_seconds():.01f} sec.")  #
    print(f"NetCDF4 data dimensions: {nc_dims}")
    print(
        f"HDF5 files discarded due to increased cloudcover: {len(h5_files)-nc_dims[0]}"
    )
    print(f"Output netCDF4 saved in: {savepath}")

def main():
    """Command Line Interface"""
    parser = argparse.ArgumentParser(
        description="Make a netCDF4 file from one or more MLST LSA-001 hdf5 files.",
        epilog="CAUTION: the input file dir should contain ONLY hdf5 files.",
    )

    parser.add_argument(
        "indir", type=str, help="the path where the input hdf5 files are stored."
    )
    parser.add_argument(
        "savedir",
        type=str,
        help="the path where the output NETCDF4 file will be stored.",
    )
    parser.add_argument(
        "nc_name", type=str, help="the filename of the output NETCDF4 file."
    )
    parser.add_argument(
        "bbox",
        type=float,
        nargs=4,
        help="the bounding box coordinates (latN, latS, lonW, lonE).",
    )
    parser.add_argument(
        "-r",
        "--res",
        metavar="X.XX",
        nargs=2,
        type=float,
        default=[0.05, 0.05],
        help="the horizontal and vertical resolution of the resampled data.",
    )

    args = parser.parse_args()

    LSALSTstack2NetCDF(
        args.indir,
        args.savedir,
        args.nc_name,
        latN=args.bbox[0],
        latS=args.bbox[1],
        lonW=args.bbox[2],
        lonE=args.bbox[3],
        hres=args.res[0],
        vres=args.res[1],
    )

if __name__ == "__main__":
    main()
