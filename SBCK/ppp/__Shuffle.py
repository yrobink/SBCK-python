
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

from .__PrePostProcessing import PrePostProcessing
from ..stats.__shuffle import MVRanksShuffle
from ..stats.__shuffle import MVQuantilesShuffle


############
## Typing ##
############

from typing import Any
from typing import Sequence
_Array = np.ndarray


#############
## Classes ##
#############

class Shuffle(PrePostProcessing):##{{{
    """Shuffle pre post processing for R2D2
    Can be used with any others BC method
    """
    
    _shuffle: MVRanksShuffle | MVQuantilesShuffle
    start_by_margins: bool
    
    def __init__( self , *args: Any , col_cond: Sequence[int] = [0] , lag_search: int = 1 , lag_keep: int = 1 , method: str = "quantile" , start_by_margins: bool = False , **kwargs: Any ) -> None:##{{{
        """
        Parameters
        ----------
        col_cond : Sequence[int]
            Conditioning columns
        lag_search: int
            Number of lags to transform the dependence structure
        lag_keep: int
            Number of lags to keep
        bc_method: SBCK.<bc_method>
            Bias correction method
        method: str
            Shuffle method used, can be "quantile" or "rank".
        start_by_margins: bool
            If True, first apply bc_method, and after the shuffle. If False, 
            reverse this operation.
        """
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name  = "Shuffle"
        self.start_by_margins = start_by_margins
        match method:
            case "rank":
                self._shuffle = MVRanksShuffle( col_cond , lag_search , lag_keep )
            case "quantile":
                self._shuffle = MVQuantilesShuffle( col_cond , lag_search , lag_keep )
            case _:
                raise ValueError("Unknow method")
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """Apply transformation"""
        
        if self._kind == "Y0":
            self._shuffle.fit(X)
            Xt = X
        else:
            if self.start_by_margins:
                Xt = X
            else:
                Xt = self._shuffle.transform(X)
        return Xt
    ##}}}
    
    def itransform( self , Xt: _Array  ) -> _Array:##{{{
        """Apply inverse transformation"""
        
        if self.start_by_margins:
            X = self._shuffle.transform(Xt)
        else:
            X = Xt
        
        return X
    ##}}}
    
##}}}

