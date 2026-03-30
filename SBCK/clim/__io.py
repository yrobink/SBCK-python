
## Copyright(c) 2026 Yoann Robin
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

############
## Typing ##
############

from typing import Sequence

import cftime
import netCDF4

import numpy as np
import xarray as xr

from .__tools import round_hour00_time


###############
## Functions ##
###############

## save_like_input_netcdf ## {{{

def save_like_input_netcdf( X: xr.DataArray,
                     ofiles: Sequence[str],
                     ifiles: Sequence[str],
                     attrs: dict[str,str] = {},
                     output_cvar_name: callable = lambda c: f"{c}Adjust",
                     output_long_name: callable = lambda c: f"Bias Corrected {c}",
                     time_dim: str = "time",
                     cvar_dim: str = "cvar", ) -> None:
    """Function used to write `X` on a list of files `ofiles` modeled on a list
    of input files `ifiles`. Typically, a model is loaded from the list
    `ifiles`, corrected, and write on `ofiles`. All attributes from input
    files are also copied, with all dimensions and supplementary variables.

    Arguments
    ---------
    X: xr.DataArray | zr.ZXArray
        Data to write. Can be a ZXArray
    ofiles: Sequence[str]
        List of files to write
    ifiles: Sequence[str]
        List of files to used as a model
    attrs: dict[str,str]
        Global attributes to add
    output_cvar_name: Callable
        Function to change the cvar name, default is to add the suffix "Adjust"
    output_long_name: Callable
        Function to change the long_name in netcdf file, default is to add the prefix 'Bias Corrected'
    time_dim: str
        Name of time axis, default is "time"
    cvar_dim: str
        Name of climate variable stacked axis, default is `cvar`
    
    Returns
    -------
    None
    """
    ##
    year0 = X[time_dim].dt.year.values[ 0]
    year1 = X[time_dim].dt.year.values[-1]

    ##
    for ifile,ofile in zip(ifiles,ofiles):

        ## Read input
        with netCDF4.Dataset( ifile, "r" ) as incf:
            
            ## Find the time axis
            i_time  = np.array(cftime.num2date( incf.variables[time_dim] , incf.variables[time_dim].units , incf.variables[time_dim].calendar ))
            i_time0 = round_hour00_time(i_time)
            i_time0 = xr.DataArray( i_time0, dims = [time_dim], coords = [i_time0] )
            o_time0 = i_time0.sel( **{ time_dim: slice(str(year0),str(year1) ) } )
            if o_time0.size == 0:
                continue
            idx_time_io = np.searchsorted( i_time0, o_time0 )
            o_time  = i_time[idx_time_io]
            
            ## Find the output variable
            cvar = list(set(X[cvar_dim].values.tolist()) & set(incf.variables))[0]

            ## Start to write the output file
            with netCDF4.Dataset( ofile, "w" ) as oncf:

                oncf.set_fill_off()
                
                ## Copy main attributes
                for name in incf.ncattrs():
                    try:
                        oncf.setncattr( name , incf.getncattr(name) )
                    except AttributeError:
                        pass
                
                ## Add others global attributes
                for key in attrs:
                    oncf.setncattr( key, attrs[key] )
                
                ## Start with dimensions
                dims   = [d for d in incf.dimensions]
                ncdims = { d : oncf.createDimension( d  , incf.dimensions[d].size )  for d in dims if not d == time_dim }
                if incf.dimensions[time_dim].isunlimited(): ## Unlimited dimensions
                    ncdims[time_dim] = oncf.createDimension( time_dim  , None )
                else:
                    ncdims[time_dim] = oncf.createDimension( time_dim  , o_time.size )
                
                ## Define variables of dimensions
                ncv_dims = {}
                for d in dims:
                    if not d in incf.variables:
                        continue
                    chk    = incf.variables[d].chunking()
                    params = { "shuffle" : False }
                    if chk == "contiguous":
                        params["contiguous"] = True
                    else:
                        params["compression"] = "zlib"
                        params["complevel"]   = 5
                        params["chunksizes"]  = chk
                    ncv_dims[d] = oncf.createVariable( d , incf.variables[d].dtype , (d,)  , **params )
                
                ## Copy attributes of the dimensions
                for d in ncv_dims:
                    for name in incf.variables[d].ncattrs():
                        try:
                            ncv_dims[d].setncattr( name , incf.variables[d].getncattr(name) )
                        except AttributeError:
                            pass
                            #print(f"Error with attribute '{name}' for dim '{d}'")
                
                ## And fill dimensions (except time_dim)
                for d in list(set(dims) & set([k for k in ncv_dims])):
                    if d == time_dim:
                        continue
                    ncv_dims[d][:] = incf.variables[d][:]
                
                ## Now fill time_dim, and add to list of dimensions
                ncv_dims[time_dim][:] = cftime.date2num( o_time , units = incf.variables[time_dim].units , calendar = incf.variables[time_dim].calendar )
                
                ## Continue with all variables, except the "main" variable
                variables = [v for v in incf.variables if v not in dims + [cvar]]
                ncvars = {}
                for v in variables:
                    
                    ## Create the variable
                    chk    = incf.variables[v].chunking()
                    params = { "shuffle" : False }
                    if chk == "contiguous":
                        params["contiguous"] = True
                    else:
                        params["compression"] = "zlib"
                        params["complevel"]   = 5
                        params["chunksizes"]  = chk
                    ncvars[v] = oncf.createVariable( v , incf.variables[v].dtype , incf.variables[v].dimensions , **params )
                    
                    ## Copy attributes
                    for name in incf.variables[v].ncattrs():
                        try:
                            ncvars[v].setncattr( name , incf.variables[v].getncattr(name) )
                        except AttributeError:
                            pass
                    
                    ## Fill it
                    if len(incf.variables[v].shape) == 0: ## Scalar variable
                        ncvars[v].assignValue(incf.variables[v].getValue())
                    else:
                        if time_dim in incf.variables[v].dimensions:
                            ## Tuple of selection
                            sel = [slice(None) for _ in range(len(incf.variables[v].dimensions))]
                            sel[list(incf.variables[v].dimensions).index(time_dim)] = idx_time_io
                            sel = tuple(sel)
                            
                            ## And copy var
                            ncvars[v][:] = incf.variables[v][sel]
                        else:
                            ncvars[v][:] = incf.variables[v][:]
                
                ## Find fill and missing value
                try:
                    fill_value = incf.variables[cvar].getncattr("_FillValue")
                except AttributeError:
                    fill_value = np.nan
                try:
                    missing_value = incf.variables[cvar].getncattr("missing_value")
                except AttributeError:
                    missing_value = None
                
                ## Create the main variable
                params = { "shuffle" : False , "fill_value" : fill_value }
                if incf.variables[cvar].chunking() == "contiguous":
                    params["contiguous"] = True
                else:
                    params["compression"] = "zlib"
                    params["complevel"]   = 5
                    params["chunksizes"]  = incf.variables[cvar].chunking()
                ncvar = oncf.createVariable( output_cvar_name(cvar) , incf.variables[cvar].dtype , incf.variables[cvar].dimensions , **params )
                if missing_value is not None:
                    ncvar.setncattr( "missing_value" , missing_value )
                
                ## Copy attributes
                for name in incf.variables[cvar].ncattrs():
                    try:
                        if name == "long_name":
                            ncvar.setncattr( name , output_long_name(incf.variables[cvar].getncattr(name)) )
                        else:
                            ncvar.setncattr( name , incf.variables[cvar].getncattr(name) )
                    except AttributeError:
                        pass
                
                ## And copy
                values = X.sel( **{ cvar_dim: cvar, time_dim: o_time0 } ).values
                ncvar[:] = np.where( np.isfinite(values), values, fill_value )
                del values
##}}}


