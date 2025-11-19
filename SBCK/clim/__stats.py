
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

import numpy as np
import xarray as xr

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

def phaversine_distances( X: np.ndarray ) -> np.ndarray:##{{{
    
    """
    SBCK.clim.phaversine_distances
    ==============================
    Also called great circle distance. This is the distance on the unit sphere
    between two lat / lon points.
    Here this is the pairwise phaversine_distances between all rows of X.

    Arguments
    ---------
    X: numpy.ndarray
        2-dimensional array where the first column is the latitude, and the
        second the longitude in radian

    Returns
    -------
    D: numpy.ndarray
        Pairwise distance matrix, with shape X.shape[0] x X.shape[0].
    """

    lat = X[:,0]
    lon = X[:,1]

    d = 2 * np.arcsin(
        np.sqrt(
            np.sin( ( lat.reshape(1,-1) - lat.reshape(-1,1) ) / 2 )**2\
            + np.cos(lat.reshape(1,-1)) * np.cos(lat.reshape(-1,1))\
            * np.sin( (lon.reshape(1,-1) - lon.reshape(-1,1)) / 2 )**2
        )
    )
    return d
##}}}

def xcorr( da_a: xr.DataArray, da_b: xr.DataArray, dim: str, method: str = "pearson" ) -> xr.DataArray:##{{{
    """
    SBCK.clim.xcorr
    ===============
    Function tu compute Pearson or Spearman cross-correlation between da_a and
    da_b. This function just call the `xarray.corr` function. If method is:
    - "pearson" : call `xarray.corr` on da_a and da_b on dim `dim`,
    - "spearman": call `xarray.corr` on da_a.rank(dim = dim) and
                da_b.rank(dim = dim) on dim `dim`
    
    Arguments
    ---------
    da_a: xarray.DataArray
        First dataarray
    da_b: xarray.DataArray
        Second dataarray
    dim: str
        Dimension along which to calculate correlations
    method: str
        "pearson" or "spearman" (rank) correlations
    
    Return
    ------
    c: xarray.DataArray
        Dataarray of correlations

    """
    match method.lower():
        case "pearson":
            c = xr.corr( da_a, da_b, dim = dim )
        case "spearman":
            c = xr.corr( da_a.rank( dim = dim ), da_b.rank( dim = dim ), dim = dim )
        case _:
            raise ValueError( f"Unknow method '{method}'")
    
    return c
##}}}

def cacorrelogram( X: xr.DataArray , lags: int | Sequence[int] = [0,3] , method: str = "pearson" ) -> xr.DataArray:##{{{
    """
    SBCK.clim.cacorrelogram
    =======================
    Function to compute the cross-auto-correlogram of X.

    Arguments
    ---------
    X: xarray.DataArray
        X must be a 4 dimensional datarray with dimensions
            ("time","cvar","lat","lon")
        the names can be differents, and cvar means climate-variable.
    lags: int | Sequence[int]
        lags to compute. Value 0 corresponds to the correlation.
    method: str
        "pearson" for classic correlation or "spearman" for rank correlations
    
    Returns
    -------
    ca: xarray.DataArray
        Cross-auto-correlogram, with dimensions:
            ("lag","month","cvar0","cvar1","distance")
        The first dimension is the lag, the second the correlation for a
        specific month, "cvar0" and "cvar1" corresponds to cross-correlation
        between two variables, and the last is the distance between two grid
        points
    """

    ## Parameters
    dim_time = X.dims[0]
    dim_cvar = X.dims[1]
    dim_lat  = X.dims[2]
    dim_lon  = X.dims[3]
    months   = [m + 1 for m in range(12)]
    nmonth   = len(months)
    cvars    = X[dim_cvar].values
    ncvar    = cvars.size
    nlatlon  = X[dim_lat].size * X[dim_lon].size

    ## Lags
    if isinstance(lags,int):
        lags = [lags]
    lags = np.array([lags]).ravel().astype(int)
    nlag = lags.size
    
    ## Chunk
    chunks   = { d : "auto" for d in X.dims[1:] }
    cX = X.chunk(chunks)

    ## Spatial anomaly
    aX = cX - cX.mean( dim = (dim_lat,dim_lon) )

    ## Split dimension
    da_a = aX.rename( { d : f"{d}0" for d in X.dims[1:] } )
    da_b = aX.rename( { d : f"{d}1" for d in X.dims[1:] } )
    
    ## Compute correlations
    corr = []
    for l in lags:
        time_a = da_a.time[l:]
        time_b = da_b.time[:-l] if l > 0 else da_b.time
        corr.append( xr.concat( [
                    xcorr( da_a.sel( time = time_a ).assign_coords( time = time_a ).groupby(f"{dim_time}.month")[m],
                             da_b.sel( time = time_b ).assign_coords( time = time_a ).groupby(f"{dim_time}.month")[m],
                             dim = dim_time,
                            method = method
                            )
                    for m in months ] , dim = "month" ).assign_coords( month = months ) )
    corr = xr.concat( corr , dim = "lag" ).assign_coords( lag = lags )
    corr = corr.compute()
    
    ## Compute pairwise distances
    lat = X[X.dims[-2]].values
    lon = X[X.dims[-1]].values
    coords_rad   = np.array([[np.radians(_lat), np.radians(_lon)] 
                       for _lat in lat for _lon in lon])
    earth_radius = 6371
    dist_km      = phaversine_distances(coords_rad) * earth_radius

    ## Reorganize
    idx_0,idx_1 = np.tril_indices(nlatlon)
    dist_km     = dist_km[idx_0,idx_1]
    sidx    = np.argsort(dist_km)
    idx_0   = idx_0[sidx]
    idx_1   = idx_1[sidx]
    dist_km = dist_km[sidx]
    
    corr = corr.transpose( "lag", "month", f"{dim_cvar}0", f"{dim_cvar}1", f"{dim_lat}0" , f"{dim_lon}0", f"{dim_lat}1", f"{dim_lon}1").values
    corr = corr.reshape( nlag, nmonth, ncvar, ncvar, nlatlon, nlatlon )
    corr = corr[:,:,:,:,idx_0,idx_1]
    corr = xr.DataArray( corr,
                         dims = ["lag","month",f"{dim_cvar}0",f"{dim_cvar}1","distance"],
                       coords = [lags,months,cvars,cvars,dist_km] )

    return corr
##}}}



