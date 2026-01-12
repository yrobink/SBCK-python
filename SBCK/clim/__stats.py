
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
import numpy as np
import xarray as xr

from ..ppp.__MomentsBC import MNPar

############
## Typing ##
############

from typing import Sequence


###############
## Functions ##
###############

######################################
## Dynamical Normalization function ##
######################################

def _dnormalization( Y0, X0, X1, timeY0, timeX0, timeX1 ):##{{{
    
    ## Output
    NX0 = X0.copy() + np.nan
    NX1 = X1.copy() + np.nan
    
    ##
    if X0.size == 1:
        return X0,X1
    
    ##
    ny,nx,ncvar,_ = Y0.shape
    cvars = range(ncvar)
    
    ## Loop
    for iy,ix,m in itt.product(range(ny),range(nx),range(12)):

        ## Extract sub-data
        idxY0 = timeY0.dt.month == m + 1
        idxX0 = timeX0.dt.month == m + 1
        idxX1 = timeX1.dt.month == m + 1
        mY0 = Y0[iy,ix,idxY0,:]
        mX0 = X0[iy,ix,idxX0,:]
        mX1 = X1[iy,ix,idxX1,:]

        ## Find parameters
        mnpY0 = MNPar(mY0)
        mnpX0 = MNPar(mX0)
        mnpX1 = MNPar(mX1)

        ## And apply normalization
        NX0[iy,ix,idxX0,:] = ( mnpY0._S @ mnpX0._ivS @ (mX0 - mnpX0._m).T ).T + mnpY0._m
        NX1[iy,ix,idxX1,:] = ( mnpX1._S @ mnpX0._ivS @ mnpY0._S @ mnpX1._ivS @ (mX1 - mnpX1._m).T ).T +\
                            mnpY0._s / mnpX0._s * ( mnpX1._m - mnpX0._m ) +\
                            mnpY0._m

    return NX0,NX1

##}}}

## dnormalization ##{{{

def dnormalization( Y0: xr.DataArray,
                    X0: xr.DataArray,
                    X1: xr.DataArray,
                   **kwargs,
                   ) -> tuple[xr.DataArray,xr.DataArray]:
    """
    SBCK.clim.dnormalization
    ========================
    Apply dynamical normalization of data to take into account of the dynamic
    between X0 and X1 to compute statistics

    Arguments
    ---------
    Y0: xarray.DataArray
        Reference in calibration period, with dimensions
        (time_name,time_cvar) + spatial_coordinates
    X0: xarray.DataArray
        Biased model in calibration period, with dimensions
        (time_name,time_cvar) + spatial_coordinates
    X1: xarray.DataArray
        Biased model in projection period, with dimensions
        (time_name,time_cvar) + spatial_coordinates
    kwargs:
        Some keywords arguments
        - "time_name": name of the time axis, default is "time"
        - "cvar_name": name of the cvar axis, default is "cvar"
        - "chunks": chunk for parallelization
        
    Returns
    -------
    NX0: xarray.DataArray
         Normalized biased model in calibration period, with dimensions
         (time_name,time_cvar) + spatial_coordinates
    NX1: xarray.DataArray
         Normalized biased model in projection period, with dimensions
         (time_name,time_cvar) + spatial_coordinates
    """
    
    ## Parameters
    time_name = kwargs.get("time_name","time")
    cvar_name = kwargs.get("cvar_name","cvar")
    
    ## Prepare
    chunks = kwargs.get("chunks")
    if chunks is None:
        chunks = { d: "auto" for d in Y0.dims[2:] }
    xY0 = Y0.rename( time = f"{time_name}Y0" ).chunk(chunks)
    xX0 = X0.rename( time = f"{time_name}X0" ).chunk(chunks)
    xX1 = X1.rename( time = f"{time_name}X1" ).chunk(chunks)
    
    ## Compute
    res = xr.apply_ufunc( _dnormalization, xY0, xX0, xX1,
                         input_core_dims  = [[f"{time_name}Y0",cvar_name],[f"{time_name}X0",cvar_name],[f"{time_name}X1",cvar_name]],
                         output_core_dims = [                             [f"{time_name}X0",cvar_name],[f"{time_name}X1",cvar_name]],
                         dask = "parallelized",
                         kwargs = { "timeY0": Y0[time_name], "timeX0": X0[time_name], "timeX1": X1[time_name] }
                         )
    res = xr.Dataset( { s: NK for s,NK in zip(["NX0","NX1"],res) } ).compute()
    
    ## Extract
    NX0 = res.NX0.rename( timeX0 = time_name ).transpose(*X0.dims)
    NX1 = res.NX1.rename( timeX1 = time_name ).transpose(*X1.dims)

    return NX0,NX1
##}}}


##########################################
## Fake climate data generator function ##
##########################################

## fakeclimdata ##{{{

def fakeclimdata( cvars: Sequence[str] = ["tas","pr"],
                 rangeY: Sequence[str] = ("1961","1980"),
                 rangeX: Sequence[str] = ("1951","2000"),
                 nlat: int = 5,
                 nlon: int = 4,
                 calendarY: str = "standard",
                 calendarX: str = "standard",
                 ):
    """
    Function used to create fake climate data. The generated data can be used
    to test the behaviour of SBCK function.

    Arguments
    ---------
    cvars: Sequence[str]
        List of names of climate variable
    rangeY: Sequence[str]
        Start and end year of reference Y
    rangeX: Sequence[str]
        Start and end year of biased data X
    nlat: int
        Numbers of latitude points
    nlon: int
        Numbers of longitude points
    calendarY: str
        Calendar of Y, see the cftime package
    calendarX: str
        Calendar of X, see the cftime package
    
    Returns
    -------
    Y: xarray.DataArray
        Reference data, with dimension ("time","cvar","lat","lon"), following
        a normal distribution N(10,0.5)
    X: xarray.DataArray
        Biased data, with dimension ("time","cvar","lat","lon"), following
        a normal distribution N(0,1)
    """
    ## Create coordinates
    timeY  = xr.date_range( f"{rangeY[0]}-01-01", f"{int(rangeY[1])+1}-01-01", use_cftime = True , calendar = calendarY )[:-1]
    timeX  = xr.date_range( f"{rangeX[0]}-01-01", f"{int(rangeX[1])+1}-01-01", use_cftime = True , calendar = calendarX )[:-1]
    lat    = np.linspace(  -90,  90, nlat )
    lon    = np.linspace( -180, 180, nlon + 1 )[1:]
    xrdims  = ("time","cvar","lat","lon")
    
    ## Random data
    Y = xr.DataArray( np.random.normal( size = (timeY.size,len(cvars),lat.size,lon.size) ) / 2 + 10 , dims = xrdims, coords = [timeY,cvars,lat,lon] )
    X = xr.DataArray( np.random.normal( size = (timeX.size,len(cvars),lat.size,lon.size) )          , dims = xrdims, coords = [timeX,cvars,lat,lon] )
    
    return Y,X
##}}}


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

def cadescribe( xca: xr.DataArray , bins: np.ndarray | None = None, q_level: Sequence[float] = [0,0.05/2,0.33,0.5,0.66,1-0.05/2,1], q_name: Sequence[str] = ["QN","QL","Q33","MED","Q66","QU","QX"] ) -> xr.Dataset:##{{{
    """
    SBCK.clim.cadescribe
    =====================
    Function to compute some statistics of a cross-auto-correlogram.

    Arguments
    ---------
    xca: xarray.DataArray
        DataArray containing the cross-auto-correlogram computed
        with SBCK.clim.cacorrelogram
    bins: np.ndarray | None
        Bins used for the distance
    q_level: Sequence[float]
        Quantile level. Default is [0,0.05/2,0.33,0.5,0.66,1-0.05/2,1]: the min
        and max, the 95% confidence interval, the 33% and 66% level, and
        the median.
    q_name: Sequence[str]
        Names of the quantile level. Default
        is ["QN","QL","Q33","MED","Q66","QU","QX"]

    Returns
    -------
    xres: xarray.Dataset
        Dataset containing 4 variables:
        - w: the weight of each bin defined by bins,
        - m: the mean in each bin,
        - s: the standard deviation of each bin
        - q: the quantile given by q_level
    """


    ## Distances
    distances  = xca["distance"].values.copy()
    if bins is None:
        db = np.diff(np.quantile( distances, [0.25,0.75] )) / 20
        bins = np.arange( 0 - db / 2, distances.max() + db / 2 + db / 4 , db )
    bdistances = (bins[1:] + bins[:-1] ) / 2
    
    ##
    xres = xr.Dataset( {
        "w": xr.DataArray( distances, dims = ["distance"], coords = [distances] ).groupby_bins( "distance" , bins = bins ).sum() / distances.size,
        "m": xca.groupby_bins( "distance" , bins = bins ).mean(),
        "s": xca.groupby_bins( "distance" , bins = bins ).std(),
        "q": xca.groupby_bins( "distance" , bins = bins ).quantile( q_level ).transpose( *(xca.dims[:-1] + ("quantile","distance_bins")) ).assign_coords( quantile = q_name ),
    } ).rename( distance_bins = "bdistance" ).assign_coords( bdistance = bdistances)

    return xres
##}}}


