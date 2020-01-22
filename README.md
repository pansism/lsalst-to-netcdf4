# lsalst-to-netcdf4
Extracts data from one or more SEVIRI LST HDF5 files, performs the geocoding and adds them to a new netCDF4. 

## Description
This script gets as input one or more MSG-SEVIRI Land Surface Temperature (LST) HDF5 products (MLST LSA-001) from EUMETSAT's Satellite Application Facility on Land Surface Analysis (LSA-SAF) and extracts, for the given bounding box and for each input file, the LST, QF and errorbar datasets. The extracted datasets are first resampled (using nearest neighbor) to a regular latitude-longitude grid and then stacked according to their acquisition time. Finally, the resulting LST, QF and errorbar stacks are added to a netCDF4 file.

MLST LSA-001 data can be downloaded from [https://landsaf.ipma.pt](https://landsaf.ipma.pt/en/products/land-surface-temperature/lst/).

## Usage 
1. Place an order on [https://landsaf.ipma.pt](https://landsaf.ipma.pt/en/products/land-surface-temperature/lst/) for SEVIRI LST data.
2. Download the data and unzip all the .bz files in an empty folder.
3. Run the script `lsalst2netcdf.py` using as arguments:
    * the directory of the unzipped HDF5 data;
    * a savedir and a savename for the output netCDF4;
    * the bounding box lat/lon coordinates; and 
    * the resolution of the output grid (optional).
    
    Example:
    ```python
   python lsalst2netcdf.py h5dir savedir savename latN, latS, lonW, lonE -r 0.05 0.05 
   ```

## Useful links
* [MLST LSA-001 Product Manual](https://landsaf.ipma.pt/GetDocument.do?id=746)
* [MLST LSA-001 Product Output Format](https://landsaf.ipma.pt/GetDocument.do?id=368)
* [MLST LSA-001 Validation Report](https://landsaf.ipma.pt/GetDocument.do?id=676)
* [MLST LSA-001 Algorithm Theoretical Basis Document](https://landsaf.ipma.pt/GetDocument.do?id=747)
* [MSG-SEVIRI](https://www.eumetsat.int/website/home/Satellites/CurrentSatellites/Meteosat/index.html)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
