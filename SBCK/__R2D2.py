# -*- coding: utf-8 -*-

## Copyright(c) 2021 / 2025 Yoann Robin
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

from .__decorators import io_fit
from .__decorators import io_predict
from .__CDFt import CDFt

from .misc.__sys import deprecated

from .ppp.__Shuffle import Shuffle


############
## Typing ##
############

from typing import Any
from typing import Sequence
_Array = np.ndarray
_NArray = _Array | None


#############
## Classes ##
#############

class R2D2(Shuffle):##{{{
    """Multivariate bias correction with quantiles shuffle, see [1].
    
    References
    ----------
    [1] Vrac, M.: Multivariate bias adjustment of high-dimensional climate
    simulations: the Rank Resampling for Distributions and Dependences (R2 D2 )
    bias correction, Hydrol. Earth Syst. Sci., 22, 3175–3196,
    https://doi.org/10.5194/hess-22-3175-2018, 2018.
    [2] Vrac, M. et S. Thao (2020). “R2 D2 v2.0 : accounting for temporal
        dependences in multivariate bias correction via analogue rank
        resampling”. In : Geosci. Model Dev. 13.11, p. 5367-5387.
        doi :10.5194/gmd-13-5367-2020.
    """
    
    def __init__( self , col_cond: Sequence[int] = [0] , lag_search: int = 1 , lag_keep: int = 1 , method: str = "quantile" , start_by_margins: bool = False , **kwargs ) -> None:##{{{
        """
        Parameters
        ----------
        col_cond : Sequence[int]
            Conditioning columns
        lag_search: int
            Number of lags to transform the dependence structure
        lag_keep: int
            Number of lags to keep
        method: str
            Shuffle method used, can be "quantile" or "rank".
        start_by_margins: bool
            If True, first apply bc_method, and after the shuffle. If False, 
            reverse this operation.
        **kwargs: ...
            all others named arguments are passed to CDFt
        """
        super().__init__( bc_method = CDFt , bc_method_kwargs = kwargs , col_cond = col_cond , lag_search = lag_search , lag_keep = lag_keep , method = method , start_by_margins = start_by_margins )
    ##}}}
##}}}


########################
## Deprecated Classes ##
########################

@deprecated( "AR2D2 code is transfered to R2R2 since the version 2.0.0" )
class AR2D2(R2D2):##{{{
    """Deprecated, use R2D2.
    """
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:
        __doc__ = R2D2.__init__.__doc__
        super().__init__( *args , **kwargs )
        self._name = "AR2D2"
    
##}}}

@deprecated( "Redundant with R2R2 since the version 2.0.0" )
class QMrs(R2D2):##{{{
    """Deprecated, use R2D2.
    """
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:
        super().__init__( *args , **kwargs )
        self._name = "QMrs"
    
    @io_fit
    def fit( self , Y0: _Array , X0: _Array ) -> _Array:
        super().fit( Y0 = Y0 , X0 = X0 , X1 = X0 )
        
        return self
    
    @io_predict
    def predict( self , X0: _Array ) -> _Array:
        return super().predict( X1 = X0 )
    
##}}}

