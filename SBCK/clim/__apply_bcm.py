
## Copyright(c) 2025, 2026 Yoann Robin
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


###############
## Libraries ##
###############

import itertools as itt
import logging

import numpy as np
import xarray as xr

from ..__AbstractBC import AbstractBC
from .__tools import yearly_window


############
## Typing ##
############

from typing import Any
from typing import Sequence


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

###############
## Functions ##
###############


def group2time( time: xr.DataArray, time_dim: str, grps: Sequence[Any], grp_name: str ) -> xr.DataArray:##{{{
    return xr.concat( [ time.groupby(f"{time_dim}.{grp_name}")[g] for g in grps ], dim = time_dim ).sortby(time_dim).compute()
##}}}


## _apply_bcm ##{{{

def _apply_bcm( Y0: np.ndarray , X0f: np.ndarray , X1f: np.ndarray,
                                 X0p: np.ndarray , X1p: np.ndarray,
               bc_method: AbstractBC,
               bc_method_kwargs: dict[str,Any] = {},
               n_multivariate_dims: int = 0 ) -> np.ndarray:
    
    ## Create output
    Z0p = X0p.copy() + np.nan
    Z1p = X1p.copy() + np.nan

    ## Special case
    if Z1p.size == 1:
        return Z1p,Z0p
    
    ## Find uni and multi-axis
    tdim  = Y0.ndim - 1 - n_multivariate_dims
    
    ## Loop
    for pidx in itt.product(*[range(s) for s in Y0.shape[:tdim]]):
        idx = pidx + (slice(None),) + tuple([slice(None) for _ in range(n_multivariate_dims)])
        shp = [1 for _ in range(len(pidx))] + [-1] + [Y0.shape[s+tdim+1] for s in range(n_multivariate_dims)]
        if not (np.isfinite(Y0[idx]).all() and
                np.isfinite(X0f[idx]).all() and
                np.isfinite(X1f[idx]).all() and
                np.isfinite(X0p[idx]).all() and
                np.isfinite(X1p[idx]).all()):
            continue
        _Z1p,_Z0p = bc_method( **bc_method_kwargs ).fit( Y0[idx] , X0f[idx] , X1f[idx] ).predict( X1p[idx], X0p[idx] )
        Z1p[idx]  = _Z1p.reshape(*shp)
        Z0p[idx]  = _Z0p.reshape(*shp)
    
    return Z1p,Z0p

##}}}

## apply_bcm ##{{{

def apply_bcm( Y0: xr.DataArray, X0: xr.DataArray, X1: xr.DataArray,
              bc_method: AbstractBC,
              time_dim: str = "time",
              seas_cycle: str = "month",
              seas_cycle_window: tuple[int,int,int] = (10,10,10),
              multivariate_dims: Sequence | str = tuple(),
              chunks: dict[str,int | str] | None = None,
              bc_method_kwargs: dict[str,Any] = {},
              **kwargs: Any
              ) -> xr.DataArray:
    """Function for correcting `X0` and `X1` biased data with the `Y0`
    reference. The `bcm` method is used, and must be a non-stationary method.
    The firs dimension must be the time axis.
    
    Arguments
    ---------
    Y0: xarray.DataArray
        Reference data in calibration period
    X0: xarray.DataArray
        Biased data in calibration period
    X1: xarray.DataArray
        Biased data in projection period
    bc_method: SBCK.AbstractBC
        Bias correction method
    time_dim: str = "time"
        Name of the time dimension
    seas_cycle: str = "month"
        How to deal with the seasonnal cycle.
        "month": correction is performed month by month
        "season": correction is performed season by season
        "window": each day is corrected with a window around of the day. The
                  length of the window is given by the
                  parameter `seas_cycle_window`
    seas_cycle_window: tuple[int,int,int] = (10,10,10)
        Window used to smooth the seasonal cycle. Noted
        wl,wc,wr = seas_cycle_window, the fit method is applied on the window
        of length wl + wc + wr, and predict is applied on centered window of
        size wc.
    multivariate_dims: tuple[str]
        Dimensions used for multivariate correction
    chunks: dict[str,int | str] | None = None
        Chunks for parallelization. Default is "auto".
    bc_method_kwargs: dict
        Keyword arguments passed to bcm
    
    Returns
    -------
    Z1: xarray.DataArray
        Corrected data in projection period
    Z0: xarray.DataArray
        Corrected data in calibration period
    """
    ## Check if time axis is in dimensions of Y and X
    if time_dim not in Y0.dims:
        raise ValueError(f"Time axis dimension '{time_dim}' is not a dimension of reference data")
    if time_dim not in X0.dims:
        raise ValueError(f"Time axis dimension '{time_dim}' is not a dimension of biased data (calibration)")
    if time_dim not in X1.dims:
        raise ValueError(f"Time axis dimension '{time_dim}' is not a dimension of biased data (projection)")
    
    ## Check how we deal with the seasonal cycle
    match seas_cycle:
        case "season":
            groups   = ["MAM","JJA","SON","DJF"]
            groups   = [ [g] for g in groups ]
            groupsXf = groups
            groupsXp = groups
            groupsY  = groups
            grp_name = "season"
        case "month":
            groups   = [ m + 1 for m in range(12) ]
            groups   = [ [g] for g in groups ]
            groupsXf = groups
            groupsXp = groups
            groupsY  = groups
            grp_name = "month"
        case "window":
            wl,wc,wr = seas_cycle_window
            ndaysX   = X0[time_dim].groupby(f"{time_dim}.dayofyear").groupers[0].size
            ndaysY   = Y0[time_dim].groupby(f"{time_dim}.dayofyear").groupers[0].size
            groupsXf = [ [ ( (d + w) % ndaysX ) + 1 for w in range(-wl, wc + wr)] for d in range(0,ndaysX,wc) ]
            groupsXp = [ [ ( (d + w) % ndaysX ) + 1 for w in range(  0, wc     )] for d in range(0,ndaysX,wc) ]
            groupsY  = [ [ ( (d + w) % ndaysY ) + 1 for w in range(-wl, wc + wr)] for d in range(0,ndaysX,wc) ]
            grp_name = "dayofyear"
        case _:
            raise ValueError(f"Unknow parameters '{seas_cycle}' for 'seas_cycle', value must be 'month', 'season' or 'window'")
    
    ## Check dimensions
    if not X0.dims == Y0.dims:
        raise ValueError("Different dimensions between X0 and Y0")
    if not X0.dims == X1.dims:
        raise ValueError("Different dimensions between X0 and X1")
    if not (X0.dims[0] == time_dim and Y0.dims[0] == time_dim and X1.dims[0] == time_dim):
        raise ValueError( f"The first fdimension of Y0, X0 and X1 must be the time (={time_dim}) dimension" )
    if isinstance( multivariate_dims , str ):
        multivariate_dims = (multivariate_dims,)
    multivariate_dims = tuple(multivariate_dims)
    for d in multivariate_dims:
        if d not in X0.dims:
            raise ValueError(f"Dimension '{d}' of multivariate_dims is not in Y0, X0 and X1")
    if time_dim in multivariate_dims:
        raise ValueError("Time dimension '{time_dim}' can not be in multivariate_dims argument")

    ## Create output
    Z1 = X1.copy() + np.nan
    Z0 = X0.copy() + np.nan
    
    ## Dask arguments
    input_core_dims  = [(tdim,) + multivariate_dims for tdim in [f"{time_dim}Y0",f"{time_dim}X0f",f"{time_dim}X1f",f"{time_dim}X0p",f"{time_dim}X1p"] ]
    output_core_dims = [(f"{time_dim}X1p",) + multivariate_dims,(f"{time_dim}X0p",) + multivariate_dims]
    
    ## Chunks
    if chunks is None:
        chunks = { d: "auto" for d in Y0.dims if d not in input_core_dims[0] + ("time",) }

    ## Loop on groups
    _fmt  = '0>{}'.format( int(np.log10(len(groupsY))+1) )
    _ngrp = len(groupsY)
    for igrps,(grpsXf,grpsXp,grpsY) in enumerate(zip(groupsXf,groupsXp,groupsY)):
        
        logger.info( f"Correction of group {igrps+1:{_fmt}} / {_ngrp}" )
        
        ## Sub-time axis for the group
        timeY0s  = group2time( Y0[time_dim], time_dim, grpsY , grp_name )
        timeX0fs = group2time( X0[time_dim], time_dim, grpsXf, grp_name )
        timeX0ps = group2time( X0[time_dim], time_dim, grpsXp, grp_name )
        timeX1fs = group2time( X1[time_dim], time_dim, grpsXf, grp_name )
        timeX1ps = group2time( X1[time_dim], time_dim, grpsXp, grp_name )
        
        ## Calibration period extraction
        Y0s  = Y0.sel( **{ time_dim: timeY0s  } ).rename( { time_dim: f"{time_dim}Y0"  } )
        X0fs = X0.sel( **{ time_dim: timeX0fs } ).rename( { time_dim: f"{time_dim}X0f" } )
        X0ps = X0.sel( **{ time_dim: timeX0ps } ).rename( { time_dim: f"{time_dim}X0p" } )
        X1fs = X1.sel( **{ time_dim: timeX1fs } ).rename( { time_dim: f"{time_dim}X1f" } )
        X1ps = X1.sel( **{ time_dim: timeX1ps } ).rename( { time_dim: f"{time_dim}X1p" } )
        
        ## Correction
        res = xr.apply_ufunc( _apply_bcm, Y0s.chunk(chunks) , X0fs.chunk(chunks) , X1fs.chunk(chunks), X0ps.chunk(chunks) , X1ps.chunk(chunks),
                          input_core_dims  = input_core_dims,
                          output_core_dims = output_core_dims,
                          dask = "parallelized",
                          kwargs = { "bc_method": bc_method , "bc_method_kwargs": bc_method_kwargs , "n_multivariate_dims": len(multivariate_dims) },
                          )
        res = xr.Dataset( { "Z1s": res[0], "Z0s": res[1] } ).compute()
        
        ## Store correction
        Z1s = res.Z1s.rename( { f"{time_dim}X1p": time_dim } ).transpose(*X1.dims)
        Z0s = res.Z0s.rename( { f"{time_dim}X0p": time_dim } ).transpose(*X0.dims)
        Z1.loc[Z1s.coords] = Z1s
        Z0.loc[Z0s.coords] = Z0s

    return Z1,Z0

##}}}


## _apply_bcm_along_time ##{{{

def _apply_bcm_along_time( Y0: np.ndarray , X0: np.ndarray , X1f: np.ndarray , X1p: np.ndarray ,
               bc_method: AbstractBC,
               bc_method_kwargs: dict[str,Any] = {},
               n_multivariate_dims: int = 0 ) -> np.ndarray:
    
    ## Create output
    Z1p = X1p.copy() + np.nan

    ## Special case
    if Z1p.size == 1:
        return Z1p
    
#    ## For dev
#    print( f"{Y0.shape} / {X0.shape} / {X1f.shape} / {X1p.shape} / {n_multivariate_dims}" )
    
    ## Find uni and multi-axis
    tdim  = Y0.ndim - 1 - n_multivariate_dims
    
    ## Loop
    for pidx in itt.product(*[range(s) for s in Y0.shape[:tdim]]):
        idx = pidx + (slice(None),) + tuple([slice(None) for _ in range(n_multivariate_dims)])
        shp = [1 for _ in range(len(pidx))] + [-1] + [Y0.shape[s+tdim+1] for s in range(n_multivariate_dims)]
        if not (np.isfinite(Y0[idx]).all() and np.isfinite(X0[idx]).all() and np.isfinite(X1f[idx]).all() and np.isfinite(X1p[idx]).all()):
            continue
        Z1p[idx] = bc_method( **bc_method_kwargs ).fit( Y0[idx] , X0[idx] , X1f[idx] ).predict( X1p[idx] ).reshape(*shp)
    
    return Z1p

##}}}

## apply_bcm_along_time ##{{{

def apply_bcm_along_time( Y: xr.DataArray, X: xr.DataArray,
              bc_method: AbstractBC,
              calibration_period: tuple[str | int,str | int],
              projection_range:tuple[str | int,str | int] | None = None,
              projection_window: tuple[int,int,int] = (5,10,5),
              time_dim: str = "time",
              seas_cycle: str = "month",
              seas_cycle_window: tuple[int,int,int] = (10,10,10),
              multivariate_dims: Sequence | str = tuple(),
              chunks: dict[str,int | str] | None = None,
              bc_method_kwargs: dict[str,Any] = {},
              **kwargs: Any
              ) -> xr.DataArray:
    """Function for correcting `X` biased data with the `Y` reference. The
    `bcm` method is used, and must be a non-stationary method. The first
    dimension must be the time axis.
    
    Arguments
    ---------
    Y: xarray.DataArray
        Reference data
    X: xarray.DataArray
        Biased data
    bc_method: SBCK.AbstractBC
        Bias correction method
    calibration_period: tuple[str|int,str|int]
        Calibration period, given by a pair of years
    projection_range: tuple[str|int,str|int] | None = None
        Period to correct, if not given all the X time axis is used
    projection_window: tuple[int,int,int] = (5,10,5)
        Format of the moving window. For (5,10,5), the `bcm` method is fitted
        on projection period on the window of size 5+10+5 = 20 years, and
        predict is applied on the central 10 years.
    time_dim: str = "time"
        Name of the time dimension
    seas_cycle: str = "month"
        How to deal with the seasonnal cycle.
        "month": correction is performed month by month
        "season": correction is performed season by season
        "window": each day is corrected with a window around of the day. The
                  length of the window is given by the
                  parameter `seas_cycle_window`
    seas_cycle_window: tuple[int,int,int] = (10,10,10)
        Window used to smooth the seasonal cycle. Noted
        wl,wc,wr = seas_cycle_window, the fit method is applied on the window
        of length wl + wc + wr, and predict is applied on centered window of
        size wc.
    multivariate_dims: tuple[str]
        Dimensions used for multivariate correction
    chunks: dict[str,int | str] | None = None
        Chunks for parallelization. Default is "auto".
    bc_method_kwargs: dict
        Keyword arguments passed to bcm
    
    Returns
    -------
    Z: xarray.DataArray
        Corrected data
    """
    ## Check if time axis is in dimensions of Y and X
    if time_dim not in Y.dims:
        raise ValueError(f"Time axis dimension '{time_dim}' is not a dimension of reference data")
    if time_dim not in X.dims:
        raise ValueError(f"Time axis dimension '{time_dim}' is not a dimension of biased data")
    
    ## Check if calibration period is available for Y and X
    cal0 = int(calibration_period[0])
    cal1 = int(calibration_period[1])
    if Y[time_dim].dt.year.min() > cal0 or Y[time_dim].dt.year.max() < cal1:
        raise ValueError(f"Calibration period '{cal0} / {cal1}' is not available for reference data")
    if X[time_dim].dt.year.min() > cal0 or X[time_dim].dt.year.max() < cal1:
        raise ValueError(f"Calibration period '{cal0} / {cal1}' is not available for biased data")
    
    ## Check the projection range
    bleft,bright = (X[time_dim].dt.year[0],X[time_dim].dt.year[-1])
    if projection_range is None:
        projection_range = (bleft,bright)
    prj0 = int(projection_range[0])
    prj1 = int(projection_range[1])
    if X[time_dim].dt.year.min() > prj0 or X[time_dim].dt.year.max() < prj1:
        raise ValueError(f"Projection period '{prj0} / {prj1}' is not available for biased data")
    
    ## Check the moving window parameters
    try:
        if not len(projection_window) == 3:
            raise ValueError
        for w in projection_window:
            if not isinstance( w , int ):
                raise ValueError
        wl,wm,wr = projection_window
    except Exception as e:
        raise ValueError(f"Incoherent projection_window '{projection_window}' {e}")
    
    ## Check how we deal with the seasonal cycle
    match seas_cycle:
        case "season":
            groups   = ["MAM","JJA","SON","DJF"]
            groups   = [ [g] for g in groups ]
            groupsXf = groups
            groupsXp = groups
            groupsY  = groups
            grp_name = "season"
        case "month":
            groups   = [ m + 1 for m in range(12) ]
            groups   = [ [g] for g in groups ]
            groupsXf = groups
            groupsXp = groups
            groupsY  = groups
            grp_name = "month"
        case "window":
            wl,wc,wr = seas_cycle_window
            ndaysX   = X[time_dim].groupby(f"{time_dim}.dayofyear").groupers[0].size
            ndaysY   = Y[time_dim].groupby(f"{time_dim}.dayofyear").groupers[0].size
            groupsXf = [ [ ( (d + w) % ndaysX ) + 1 for w in range(-wl, wc + wr)] for d in range(0,ndaysX,wc) ]
            groupsXp = [ [ ( (d + w) % ndaysX ) + 1 for w in range(  0, wc     )] for d in range(0,ndaysX,wc) ]
            groupsY  = [ [ ( (d + w) % ndaysY ) + 1 for w in range(-wl, wc + wr)] for d in range(0,ndaysX,wc) ]
            grp_name = "dayofyear"
        case _:
            raise ValueError(f"Unknow parameters '{seas_cycle}' for 'seas_cycle', value must be 'month', 'season' or 'window'")
    
    ## Check dimensions
    if not X.dims == Y.dims:
        raise ValueError("Different dimensions between X and Y")
    if not (X.dims[0] == time_dim and Y.dims[0] == time_dim):
        raise ValueError( f"The first fdimension of X and Y must be the time (={time_dim}) dimension" )
    if isinstance( multivariate_dims , str ):
        multivariate_dims = (multivariate_dims,)
    multivariate_dims = tuple(multivariate_dims)
    for d in multivariate_dims:
        if d not in X.dims:
            raise ValueError(f"Dimension '{d}' of multivariate_dims is not in X")
    if time_dim in multivariate_dims:
        raise ValueError("Time dimension '{time_dim}' can not be in multivariate_dims argument")

    ## Create output
    Z = X.copy() + np.nan
    
    ## Dask arguments
    input_core_dims  = [(tdim,) + multivariate_dims for tdim in [f"{time_dim}Y0",f"{time_dim}X0",f"{time_dim}X1f",f"{time_dim}X1p"] ]
    output_core_dims = [(f"{time_dim}X1p",) + multivariate_dims]
    
    ## Chunks
    if chunks is None:
        chunks = { d: "auto" for d in Y.dims if d not in input_core_dims[0] + ("time",) }
    
    ## Time axis
    timeY0 = Y[time_dim].sel( { time_dim: slice(str(cal0),str(cal1)) } )
    timeX0 = X[time_dim].sel( { time_dim: slice(str(cal0),str(cal1)) } )
    timeX  = X[time_dim]
    
    ## Loop on groups
    _fmt  = '0>{}'.format( int(np.log10(len(groupsY))+1) )
    _ngrp = len(groupsY)
    for igrps,(grpsXf,grpsXp,grpsY) in enumerate(zip(groupsXf,groupsXp,groupsY)):
        
        logger.info( f"Correction of group {igrps+1:{_fmt}} / {_ngrp}" )
        
        ## Sub-time axis for the group
        timeY0s  = group2time( timeY0, time_dim, grpsY , grp_name )
        timeX0s  = group2time( timeX0, time_dim, grpsXf, grp_name )
        timeX1fs = group2time( timeX , time_dim, grpsXf, grp_name )
        timeX1ps = group2time( timeX , time_dim, grpsXp, grp_name )
        
        ## Calibration period extraction
        Y0s = Y.sel( **{ time_dim: timeY0s } ).rename( { time_dim: f"{time_dim}Y0" } )
        if not Y0s.size > 0:
            logger.warning( f"0-size for Y0 (grp: {grpsY})" )
            continue
        X0s = X.sel( **{ time_dim: timeX0s } ).rename( { time_dim: f"{time_dim}X0" } )
        if not X0s.size > 0:
            logger.warning( f"0-size for X0 (grp: {grpsXf})" )
            continue
        
        ## Loop on years
        for tf0,tp0,tp1,tf1 in yearly_window( prj0 , prj1 , wl , wm , wr , bleft , bright ):
            
            ## Sub-time axis for the projection period
            _timeX1fs = timeX1fs.sel( { time_dim : slice(str(tf0),str(tf1)) } ).sortby(time_dim)
            _timeX1ps = timeX1ps.sel( { time_dim : slice(str(tp0),str(tp1)) } ).sortby(time_dim)
            
            ## Data extraction
            X1fs = X.sel( **{ time_dim: _timeX1fs } ).rename( { time_dim: f"{time_dim}X1f" } )
            if not X1fs.size > 0:
                logger.warning( f"0-size for X1fs (grp: {grpsXf}, per: {tf0} / {tp0} / {tp1} / {tf1})" )
                continue
            X1ps = X.sel( **{ time_dim: _timeX1ps } ).rename( { time_dim: f"{time_dim}X1p" } )
            if not X1ps.size > 0:
                logger.warning( f"0-size for X1ps (grp: {grpsXp}, per: {tf0} / {tp0} / {tp1} / {tf1})" )
                continue
            
            ## Correction
            Z1ps = xr.apply_ufunc( _apply_bcm_along_time, Y0s.chunk(chunks) , X0s.chunk(chunks) , X1fs.chunk(chunks) , X1ps.chunk(chunks) ,
                                  input_core_dims  = input_core_dims,
                                  output_core_dims = output_core_dims,
                                  dask = "parallelized",
                                  kwargs = { "bc_method": bc_method , "bc_method_kwargs": bc_method_kwargs , "n_multivariate_dims": len(multivariate_dims) },
                                  ).rename( { f"{time_dim}X1p" : time_dim } ).transpose(*X.dims).compute()
            
            ## Store correction
            Z.loc[Z1ps.coords] = Z1ps
    
    ## Final sub-selection
    Z = Z.sel( { time_dim : slice(str(prj0),str(prj1)) } )

    return Z

##}}}



