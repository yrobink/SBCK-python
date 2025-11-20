
## Copyright(c) 2025 Yoann Robin
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
import numpy as np
import xarray as xr
import zxarray as zr

from .__stats import phaversine_distances
from .__stats import xcorr


############
## Typing ##
############

from typing import Sequence


###############
## Functions ##
###############

#####################################
## Cross-auto-correlogram function ##
#####################################

def _zcorr( da_a, da_b , ndim_a, ndim_b , method = "pearson" ):##{{{

    C = xcorr(
        xr.DataArray( da_a.reshape(-1,da_a.shape[-1]) , dims = ["dim0","time"] ),
        xr.DataArray( da_b.reshape(-1,da_b.shape[-1]) , dims = ["dim1","time"] ),
        dim = "time",
        method = method
    ).values.reshape( tuple(da_a.shape[:ndim_a]) + tuple(da_b.shape[:ndim_b]) )
    
    return C
##}}}

def zcorr( da_a: zr.ZXArray, da_b: zr.ZXArray, dim: str, method: str = "pearson" , **kwargs ) -> zr.ZXArray:##{{{
    
    ## Output
    zdims = [d for d in da_a.dims if not d == dim]\
          + [d for d in da_b.dims if not d == dim]
    zcoords = [da_a[d].values for d in da_a.dims if not d == dim]\
            + [da_b[d].values for d in da_b.dims if not d == dim]

    ## Dask
    dask_kwargs = {
        "input_core_dims" : [(dim,),(dim,)],
        "output_core_dims": [[]],
        "dask": "parallelized",
        "kwargs": { "ndim_a": len(da_a.dims) - 1, "ndim_b": len(da_b.dims) - 1, "method": method }
    }
    
    ## Block memory function
    key = "block_memory"
    if kwargs.get(key) is None:
        nbits = zr.DMUnit.nbitsof_dtype(da_a.dtype)
        sdim  = da_a[dim].size
        kwargs[key] = lambda b : 5 * sdim * np.prod(b) * zr.DMUnit( n = nbits // zr.DMUnit.bits_per_octet , unit = 'o' )
    
    ## Compute
    zc = zr.apply_ufunc( _zcorr, da_a, da_b,
                         block_dims = zdims,
                         output_dims = [zdims],
                         output_coords = [zcoords],
                         dask_kwargs = dask_kwargs,
                         **kwargs )

    return zc
##}}}

def zcacorrelogram( zX: zr.ZXArray , lags: int | Sequence[int] = (0,3) , method: str = "pearson" , **kwargs ) -> zr.ZXArray:##{{{
    """
    zcacorrelogram
    ==============
    Function to compute the cross-auto-correlogram of X, but with zxarray to
    limit the memory used

    Arguments
    ---------
    X: zxarray.ZXArray
        X must be a 4 dimensional ZXArray with dimensions
            ("time","cvar","lat","lon")
        the names can be differents, and cvar means climate-variable.
    lags: int | Sequence[int]
        lags to compute. Value 0 corresponds to the correlation.
    method: str
        "pearson" for classic correlation or "spearman" for rank correlations
    kwargs:
        See `zxarray.apply_ufunc` arguments for memory and workers
    
    Returns
    -------
    ca: zxarray.ZXArray
        Cross-auto-correlogram, with dimensions:
            ("lag","month","cvar0","cvar1","distance")
        The first dimension is the lag, the second the correlation for a
        specific month, "cvar0" and "cvar1" corresponds to cross-correlation
        between two variables, and the last is the distance between two grid
        points
    """
    
    ## Parameters
    dim_time = zX.dims[0]
    dim_cvar = zX.dims[1]
    dim_lat  = zX.dims[2]
    dim_lon  = zX.dims[3]
    months   = [m + 1 for m in range(12)]
    
    ## Lags
    if isinstance(lags,int):
        lags = [lags]
    lags = np.array([lags]).ravel().astype(int)

    ## Spatial anomaly
    zA = zX.anomaly( dims = (dim_lat,dim_lon) , **kwargs )
    
    ## Split dimension
    zA0 = zA.copy().rename( **{ d : f"{d}0" for d in zA.dims[1:] } )
    zA1 = zA.copy().rename( **{ d : f"{d}1" for d in zA.dims[1:] } )

    ## Compute correlations
    xrdims = ["lag","month"] + [f"{d}0" for d in zX.dims[1:]]\
                             + [f"{d}1" for d in zX.dims[1:]]
    xrcoords = [lags,months] + [zX[d].values for d in zX.dims[1:]]\
                             + [zX[d].values for d in zX.dims[1:]]
    zc = zr.ZXArray( dims = xrdims, coords = xrcoords )
    for l,m in itt.product(lags,months):
        
        ## Extract
        time   = zX[dim_time]
        time_a = time[l:]
        time_b = time[:-l] if l > 0 else time
        da_a = zA0.zsel( **{ dim_time: time_a } ).assign_coords( **{dim_time: time_a} ).zsel( **{ dim_time: time_a.groupby(f"{dim_time}.month")[m] } )
        da_b = zA1.zsel( **{ dim_time: time_b } ).assign_coords( **{dim_time: time_a} ).zsel( **{ dim_time: time_a.groupby(f"{dim_time}.month")[m] } )
        
        ## Compute
        zc.zloc[l,m,:,:,:,:,:,:] = zcorr(
                     da_a = da_a,
                     da_b = da_b,
                     dim = dim_time,
                     method = method,
                     **kwargs
                     )
    
    ## Compute pairwise distances
    lat = zX[zX.dims[-2]].values
    lon = zX[zX.dims[-1]].values
    coords_rad   = np.array([[np.radians(_lat), np.radians(_lon)] 
                       for _lat in lat for _lon in lon])
    earth_radius = 6371
    dist_km      = phaversine_distances(coords_rad) * earth_radius
    idx_0,idx_1  = np.tril_indices(dist_km.shape[0])
    dist_km      = dist_km[(idx_0,idx_1)]
    
    sidx = np.argsort(dist_km)

    dist_km = dist_km[sidx]
    idx_0   = idx_0[sidx]
    idx_1   = idx_1[sidx]
    
    zz = zr.ZXArray( dims  = ["lag","month",f"{dim_cvar}0",f"{dim_cvar}1","distance"],
                    coords = [lags,months,zc[f"{dim_cvar}0"],zc[f"{dim_cvar}1"],dist_km]
    )

    for k in range(dist_km.size):
        
        k0 = idx_0[k]
        k1 = idx_1[k]
        j0 = k0 // lon.size
        i0 = k0  % lon.size
        j1 = k1 // lon.size
        i1 = k1  % lon.size
        zz[:,:,:,:,k] = zc[:,:,:,j0,i0,:,j1,i1]

    return zz
##}}}

