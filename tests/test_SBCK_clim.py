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
import itertools as itt

import numpy as np
import xarray as xr

import SBCK as bc
import zxarray as zr

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
        
        self.calendars = ["standard","noleap","360_day"]
    ##}}}
    
    def _data_generator( self, calY = "standard" , calX = "standard" ):##{{{
        return bc.clim.fakeclimdata( rangeY = ("1961","1965"),
                                     rangeX = ("1961","1970"),
                                    calendarY = calY,
                                    calendarX = calX,
                                    )
    ##}}}
    
    
    def _test_apply( self, Y0, X0, X1, multivariate_dims = tuple() ):##{{{
        
        ## Correction with xarray
        Z1,Z0 = bc.clim.apply_bcm( Y0, X0, X1,
                              bc_method = bc.dOTC1d,
                              seas_cycle = "season",
                              multivariate_dims = multivariate_dims,
                              )
        
        ## Transform in zxarray
        zY0 = zr.ZXArray.from_xarray(Y0)
        zX0 = zr.ZXArray.from_xarray(X0)
        zX1 = zr.ZXArray.from_xarray(X1)
        
        ## Correction with zxarray
        zZ1,zZ0 = bc.clim.zapply_bcm( zY0, zX0, zX1,
                              bc_method = bc.dOTC1d,
                              seas_cycle = "season",
                              multivariate_dims = multivariate_dims,
                              )
        
        ## Test equality
        self.assertAlmostEqual( np.abs(zZ1.dataarray - Z1).values.max(), 0 )
        self.assertAlmostEqual( np.abs(zZ0.dataarray - Z0).values.max(), 0 )
    ##}}}
    
    def test_apply_nospatial_univariate( self ):##{{{
        
        for calY,calX in itt.product(self.calendars,self.calendars):
            Y,X = self._data_generator( calY = calY, calX = calX )
            
            Y0 = Y.sel( time = slice("1961","1965") )[:,:,0,0].drop_vars(["lat","lon"])
            X0 = X.sel( time = slice("1961","1965") )[:,:,0,0].drop_vars(["lat","lon"])
            X1 = X.sel( time = slice("1966","1970") )[:,:,0,0].drop_vars(["lat","lon"])
            
            self._test_apply( Y0, X0, X1, multivariate_dims = tuple() )

    ##}}}

    def test_apply_nospatial_multivariate( self ):##{{{
        
        for calY,calX in itt.product(self.calendars,self.calendars):
            Y,X = self._data_generator( calY = calY, calX = calX )
            
            Y0 = Y.sel( time = slice("1961","1965") )[:,:,0,0].drop_vars(["lat","lon"])
            X0 = X.sel( time = slice("1961","1965") )[:,:,0,0].drop_vars(["lat","lon"])
            X1 = X.sel( time = slice("1966","1970") )[:,:,0,0].drop_vars(["lat","lon"])
            
            self._test_apply( Y0, X0, X1 , multivariate_dims = "cvar" )
    ##}}}
    
    def test_apply_spatial_univariate( self ):##{{{
        
        for calY,calX in itt.product(self.calendars,self.calendars):
            Y,X = self._data_generator( calY = calY, calX = calX )
            
            Y0 = Y.sel( time = slice("1961","1965") )
            X0 = X.sel( time = slice("1961","1965") )
            X1 = X.sel( time = slice("1966","1970") )
            
            self._test_apply( Y0, X0, X1 , multivariate_dims = "cvar" )

    ##}}}

    def test_apply_spatial_multivariate( self ):##{{{
        
        for calY,calX in itt.product(self.calendars,self.calendars):
            Y,X = self._data_generator( calY = calY, calX = calX )
            
            Y0 = Y.sel( time = slice("1961","1965") )
            X0 = X.sel( time = slice("1961","1965") )
            X1 = X.sel( time = slice("1966","1970") )
            
            self._test_apply( Y0, X0, X1 , multivariate_dims = "cvar" )

    ##}}}
    
    
    def _test_apply_along_time( self, Y, X, multivariate_dims = tuple() ):##{{{
        
        ## Correction with xarray
        Z = bc.clim.apply_bcm_along_time( Y, X,
                              bc_method = bc.dOTC1d,
                              calibration_period = ("1961","1965"),
                              projection_range = ("1961","1970"),
                              projection_window = (1,5,1),
                              seas_cycle = "season",
                              multivariate_dims = multivariate_dims,
                              )
        
        ## Transform in zxarray
        zY = zr.ZXArray.from_xarray(Y)
        zX = zr.ZXArray.from_xarray(X)
        
        ## Correction with zxarray
        zZ = bc.clim.zapply_bcm_along_time( zY, zX,
                              bc_method = bc.dOTC1d,
                              calibration_period = ("1961","1965"),
                              projection_range = ("1961","1970"),
                              projection_window = (1,5,1),
                              seas_cycle = "season",
                              multivariate_dims = multivariate_dims,
                              )
        
        ## Test equality
        self.assertAlmostEqual( np.abs(zZ.dataarray - Z).values.max(), 0 )
    ##}}}
    
    def test_apply_nospatial_univariate_along_time( self ):##{{{
        
        for calY,calX in itt.product(self.calendars,self.calendars):
            Y,X = self._data_generator( calY = calY, calX = calX )
            
            Y = Y[:,:,0,0].drop_vars(["lat","lon"])
            X = X[:,:,0,0].drop_vars(["lat","lon"])
            
            self._test_apply_along_time( Y, X, multivariate_dims = tuple() )

    ##}}}

    def test_apply_nospatial_multivariate_along_time( self ):##{{{
        
        for calY,calX in itt.product(self.calendars,self.calendars):
            Y,X = self._data_generator( calY = calY, calX = calX )
            
            Y = Y[:,:,0,0].drop_vars(["lat","lon"])
            X = X[:,:,0,0].drop_vars(["lat","lon"])
            
            self._test_apply_along_time( Y, X, multivariate_dims = "cvar" )

    ##}}}

    def test_apply_spatial_univariate_along_time( self ):##{{{
        
        for calY,calX in itt.product(self.calendars,self.calendars):
            Y,X = self._data_generator( calY = calY, calX = calX )
            
            self._test_apply_along_time( Y, X, multivariate_dims = tuple() )

    ##}}}

    def test_apply_spatial_multivariate_along_time( self ):##{{{
        
        for calY,calX in itt.product(self.calendars,self.calendars):
            Y,X = self._data_generator( calY = calY, calX = calX )
            
            self._test_apply_along_time( Y, X, multivariate_dims = "cvar" )

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

