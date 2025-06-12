
## Copyright(c) 2023 / 2025 Yoann Robin
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

from .__PrePostProcessing import PrePostProcessing
from ..misc.__linalg import as2d
from ..misc.__sys import deprecated

import warnings
import numpy as np


############
## Typing ##
############

from typing import Sequence
from typing import Any

_Array = np.ndarray
_Cols = Sequence[int] | int | None


#############
## Classes ##
#############

class FilterWarnings(PrePostProcessing):##{{{
    """This PPP method is used to supress all warnings raised by python during
    the execution
    """
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:
        """
        Constructor
        ===========
        
        Arguments
        ---------
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        *kwargs:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        """
        warnings.simplefilter("ignore")
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name = "FilterWarnings"
##}}}

class Xarray(PrePostProcessing):###{{{
    """This PPP method is used to deal with xarray. The xarray interface is
    removed before the fit, and applied to output of predict method.
    """
    
    _xcls: type | None
    _sX0: dict[bool,Any]
    _sX1: dict[bool,Any]

    def __init__( self , *args: Any , **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        *kwargs:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        """
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name = "Xarray"
        self._xcls = None
        self._sX0  = {}
        self._sX1  = {}
    
    def transform( self , X: _Array ) -> _Array:
        
        self._xcls = type(X)
        if self._kind == 'X0':
            self._sX0["dims"]   = X.dims
            self._sX0["coords"] = X.coords
            self._sX0["shape"]  = X.shape
        if self._kind == 'X1':
            self._sX1["dims"]   = X.dims
            self._sX1["coords"] = X.coords
            self._sX1["shape"]  = X.shape
        
        Xt = X.values
        return Xt
    
    def itransform( self , Xt: _Array ) -> _Array:
        
        if self._kind == "X0":
            X = self._xcls( Xt.reshape( self._sX0["shape"] ) , dims = self._sX0["dims"] , coords = self._sX0["coords"] )
        elif self._kind == "X1":
            X = self._xcls( Xt.reshape( self._sX1["shape"] ) , dims = self._sX1["dims"] , coords = self._sX1["coords"] )
        else:
            X = Xt
        
        return X
##}}}


class As2d(PrePostProcessing):##{{{
    """
    This PPP method is used to transform input in 2d array. All dimensions
    except the first are flatten. The predict method keep the shape.
    """
    
    _shape: dict[str,tuple[int,...]]

    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        """
        Arguments
        ---------
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        *kwargs:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        """
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name  = "As2d"
        self._shape = {}
    ##}}}
    
    def transform( self , X ):##{{{
        self._shape[self._kind] = X.shape
        return as2d(X)
    ##}}}
    
    def itransform( self , Xt ):##{{{
        return Xt.reshape(self._shape[self._kind])
    ##}}}
    
#}}}


######################
## Deprecated names ##
######################

@deprecated( "PPPIgnoreWarnings is renamed FilterWarnings since the version 2.0.0" )
class PPPIgnoreWarnings(FilterWarnings):##{{{
    
    def __init__( self , *args , **kwargs ):##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPIgnoreWarnings"
    ##}}}
    
##}}}

@deprecated( "PPPXarray is renamed Xarray since the version 2.0.0" )
class PPPXarray(Xarray):##{{{
    
    def __init__( self , *args , **kwargs ):##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPXarray"
    ##}}}
    
##}}}




