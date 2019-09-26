# -*- coding: utf-8 -*-

import unittest
from context import lsalst2netcdf


class Testlsalst2netcdf(unittest.TestCase):

    SAF_MSGdisk = {
        "SUB_SATELLITE_POINT_START_LON": 0.0,
        "COFF": 1857,
        "LOFF": 1857,
        "CFAC": 13642337,
        "LFAC": 13642337,
        "NC": 3712,
        "NL": 3712,
    }

    grid = {"north_lat": 45.3, "south_lat": 35.2, "west_lon": 18.2, "east_lon": 30.1}

    def test_latlon2rowcol(self):
        result = lsalst2netcdf.latlon2rowcol(
            lat=45.3, lon=-18.2, product_attrs=Testlsalst2netcdf.SAF_MSGdisk
        )
        self.assertEqual(result, (1415, 445))

        # Test for zero lat, lon
        result = lsalst2netcdf.latlon2rowcol(
            lat=0.0, lon=0.0, product_attrs=Testlsalst2netcdf.SAF_MSGdisk
        )
        self.assertEqual(result, (1857, 1857))

    def test__BBox2RowColGrid(self):
        row_grid, col_grid = lsalst2netcdf._BBox2RowColGrids(
            bbox=Testlsalst2netcdf.grid,
            hres=0.05,
            vres=0.05,
            product_attrs=Testlsalst2netcdf.SAF_MSGdisk,
        )
        self.assertEqual(row_grid.shape, (202, 239))
        self.assertEqual(sum(row_grid.flatten()), 27494577)
        self.assertEqual(sum(col_grid.flatten()), 119948695)


if __name__ == "__main__":
    unittest.main()
