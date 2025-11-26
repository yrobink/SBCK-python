
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

import logging
import gc
import psutil
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

##################
## Init logging ##
##################

logging.captureWarnings(True)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    cvars    = zX[dim_cvar]

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
    logger.info("Start lags / months cross-product")
    xrdims = ["lag","month"] + [f"{d}0" for d in zX.dims[1:]]\
                             + [f"{d}1" for d in zX.dims[1:]]
    xrcoords = [lags,months] + [zX[d].values for d in zX.dims[1:]]\
                             + [zX[d].values for d in zX.dims[1:]]
    zc = zr.ZXArray( dims = xrdims, coords = xrcoords )
    logger.info( " * Temporary file initialized:" )
    logger.info( f" * {zc}" )
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
        break
    gc.collect()
    
    ## Compute pairwise distances
    logger.info("Compute pairwise distances")
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
    del coords_rad
    gc.collect()
    
    logger.info( "Init output file" )
    zz = zr.ZXArray( dims  = ["lag","month",f"{dim_cvar}0",f"{dim_cvar}1","distance"],
                    coords = [lags,months,zc[f"{dim_cvar}0"],zc[f"{dim_cvar}1"],dist_km]
    )
    logger.info( f" * {zz}" )
    
    ## Find total memory available
    total_memory = kwargs.get("total_memory")
    memory_per_worker = kwargs.get("memory_per_worker")
    n_workers = kwargs.get("n_workers")
    if total_memory is not None and memory_per_worker is not None:
        raise ValueError( "total_memory and memory_per_worker can not be set simultaneously" )
    if total_memory is None and memory_per_worker is None:
        total_memory = zr.DMUnit( n = int( 0.8 * psutil.virtual_memory().total ) , unit = 'B' )
    if memory_per_worker is not None:
        memory_per_worker = zr.DMUnit(memory_per_worker)
        total_memory = n_workers * memory_per_worker
    else:
        total_memory = zr.DMUnit(total_memory)
    
    ## Find step to minimize operations
    logger.info("Find step size")
    max_mem_used = zr.DMUnit.sizeof_array(zz)
    min_mem_used = max_mem_used // dist_km.size
    
    if 3 * max_mem_used < total_memory:
        step = dist_km.size
    elif 3 * min_mem_used < total_memory:
        step = min( int(np.floor( np.pow( total_memory.b // (3 * min_mem_used.b ) , 1 / 4 ) )), dist_km.size )
    else:
        raise MemoryError("Not enough memory available")
    logger.info( f" * Step size found: {step}, {dist_km.size // step} copy required" )
    
    ##
    idx_lags   = np.arange( 0, len(lags)   , 1 ).astype(int).tolist()
    idx_months = np.arange( 0, len(months) , 1 ).astype(int).tolist()
    idx_cvars  = np.arange( 0, len(cvars)  , 1 ).astype(int).tolist()
    ij         = [i for i in range(step)]
    for k in range(0,dist_km.size,step):
        
        ks = slice( k, k + step, 1 )
        k0 = idx_0[ks]
        k1 = idx_1[ks]
        j0 = k0 // lon.size
        i0 = k0  % lon.size
        j1 = k1 // lon.size
        i1 = k1  % lon.size
        args = (idx_lags,idx_months,idx_cvars,j0.tolist(),i0.tolist(),idx_cvars,j1.tolist(),i1.tolist())
        S = np.prod([len(idx) for idx in args])
        logger.info( f"S: {S}" )
        T = zc.isel( drop = False, **{ d: idx for d,idx in zip(zc.dims,args) } ).values
        T = T[:,:,:,ij,ij,:,ij,ij].transpose(1,2,3,4,0)
        zz[:,:,:,:,ks] = T
#        zz[:,:,:,:,ks] = zc._internal.zdata.oindex[*args][:,:,:,ij,ij,:,ij,ij].transpose(1,2,3,4,0)

    return zz
##}}}

