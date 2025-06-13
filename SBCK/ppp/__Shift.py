
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

from .__PrePostProcessing import PrePostProcessing

import numpy as np

from ..stats.__shift import Shift as _Shift

############
## Typing ##
############

from typing import Any
_Array = np.ndarray


#############
## Classes ##
#############

class Shift(PrePostProcessing):##{{{
    """Shift pre post processing for TSMBC / dTSMBC
    Can be used with any others BC method
    """
    
    _shift: _Shift

    def __init__( self , *args: Any , lag: int = 3 , method: str = "row" , ref: int | str = "middle" , **kwargs: Any ) -> None:##{{{
        """
        Parameters
        ----------
        lag    : integer
            Time lag of the shift
        method : string
            Inverse method, "row" or "col"
        ref    : integer
            Reference columns / rows to inverse
        """
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name  = "Shift"
        self._shift = _Shift( lag = lag , method = method , ref = ref )
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """Apply transformation"""
        Xt = self._shift.transform(X)
        return Xt
    ##}}}
    
    def itransform( self , Xt: _Array  ) -> _Array:##{{{
        """Apply inverse transformation"""
        
        X = self._shift.inverse(Xt)
        
        return X
    ##}}}
    
##}}}

