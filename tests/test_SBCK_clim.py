#!/usr/bin/env python3 -m unittest

## Copyright(c) 2024, 2025 Yoann Robin
## 
## This file is part of SBCK.
## 
## SBCK is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## SBCK is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with SBCK.  If not, see <https://www.gnu.org/licenses/>.


#############
## Imports ##
#############

import os
import sys
import distributed
import unittest
import argparse

import numpy as np
import xarray as xr

import SBCK as bc

has_mpl = True
try:
    import matplotlib as mpl
    import matplotlib.pyplot   as plt
    import matplotlib.gridspec as mplg
except Exception:
    has_mpl = False

########################
## Set mpl parameters ##
########################

try:
    mpl.rcdefaults()
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.linewidth']  = 0.5
    mpl.rcParams['lines.linewidth'] = 0.5
    mpl.rcParams['patch.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
except Exception:
    has_mpl = False


######################
## Parameters class ##
######################

class SBCKTestParameters:##{{{
    
    PLOTFIG: bool = False
    
    def __init__( self ):##{{{
        lpath = os.path.join( *os.path.basename(__file__).split(".")[0].split("_")[1:] )
        self.opath = os.path.join( os.path.dirname(__file__) , "figures" , lpath )
        if not os.path.isdir(self.opath):
            os.makedirs(self.opath)
    ##}}}
    
    @property
    def prefix(self):
        return str(type(self)).split("'")[1].split(".")[-1]
    
##}}}


class Test_Apply(SBCKTestParameters,unittest.TestCase):##{{{

    def __init__( self , *args, **kwargs ):##{{{
        SBCKTestParameters.__init__( self )
        unittest.TestCase.__init__( self , *args , **kwargs )
    ##}}}
    
    def _data_generator( self, calY = "standard" , calX = "standard" ):##{{{
        timeY = xr.date_range( "1961-01-01" , "1965-12-30" , calendar = calY )
        timeX = xr.date_range( "1961-01-01" , "1970-12-30" , calendar = calX )
        cvars = ["tas","pr"]
        lat   = np.linspace(41,52,3+2)[1:-1]
        lon   = np.linspace(-5,10,4+2)[1:-1]
        Y = xr.DataArray( np.random.normal( size = timeY.size * 2 * lat.size * lon.size ).reshape(timeY.size,2,lat.size,lon.size),
                          dims = ["time","cvar","lat","lon"],
                         coords = [timeY,cvars,lat,lon]
                         )
        X = xr.DataArray( np.random.normal( size = timeX.size * 2 * lat.size * lon.size ).reshape(timeX.size,2,lat.size,lon.size),
                          dims = ["time","cvar","lat","lon"],
                         coords = [timeX,cvars,lat,lon]
                         )
        return Y,X
    ##}}}
    
    def test_apply_nospatial_univariate( self ):##{{{

        Y,X = self._data_generator( "noleap" , "360_day" )
        Y = Y[:,:,0,0].drop_vars(["lat","lon"])
        X = X[:,:,0,0].drop_vars(["lat","lon"])
        bc.clim.apply_bcm( Y, X,
                              bc_method = bc.IdBC,
                              calibration_period = ("1961","1965"),
                              projection_range = ("1961","1970"),
                              projection_window = (1,5,1),
                              seas_cycle = "season",
                              )
    ##}}}

    def test_apply_nospatial_multivariate( self ):##{{{

        Y,X = self._data_generator( "noleap" , "360_day" )
        Y = Y[:,:,0,0].drop_vars(["lat","lon"])
        X = X[:,:,0,0].drop_vars(["lat","lon"])
        bc.clim.apply_bcm( Y, X,
                              bc_method = bc.IdBC,
                              calibration_period = ("1961","1965"),
                              projection_range = ("1961","1970"),
                              projection_window = (1,5,1),
                              seas_cycle = "season",
                              multivariate_dims = "cvar",
                              )
    ##}}}

    def test_apply_multivariate( self ):##{{{

        Y,X = self._data_generator( "noleap" , "360_day" )
        bc.clim.apply_bcm( Y, X,
                              bc_method = bc.IdBC,
                              calibration_period = ("1961","1965"),
                              projection_range = ("1961","1970"),
                              projection_window = (1,5,1),
                              seas_cycle = "season",
                              multivariate_dims = "cvar",
                              )
    ##}}}

    def test_apply_window( self ):##{{{

        Y,X = self._data_generator( "noleap" , "360_day" )
        bc.clim.apply_bcm( Y, X,
                              bc_method = bc.IdBC,
                              calibration_period = ("1961","1965"),
                              projection_range = ("1961","1970"),
                              projection_window = (1,5,1),
                              seas_cycle = "window",
    #                          multivariate_dims = "cvar",
                              )
    ##}}}
    
##}}}

##########
## main ##
##########

if __name__ == "__main__":
    ## Custom parser to pass arguments for figures
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures', action = "store_true" )
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args
    SBCKTestParameters.PLOTFIG = args.figures and has_mpl
    
    ## Init dask
    ncpu = max( 1 , int(0.5 * os.cpu_count()) )
    meml = "{}GB".format( int( 0.2 * distributed.system.psutil.virtual_memory().total / 10**9 / ncpu ) )
    cluster = distributed.LocalCluster( n_workers = ncpu, memory_limit = meml , processes = False )
    client  = cluster.get_client()
    
    ## And run
    unittest.main()

