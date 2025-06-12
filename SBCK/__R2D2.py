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

from .__AbstractBC import AbstractBC
from .__decorators import io_fit
from .__decorators import io_predict
from .__CDFt import CDFt

from .misc.__sys import deprecated
from .stats.__shuffle import MVQuantilesShuffle
from .stats.__shuffle import MVRanksShuffle


############
## Typing ##
############

from typing import Any
from typing import Self
from typing import Sequence
from .__AbstractBC import _rv_type
from .__AbstractBC import _mrv_type
_Array = np.ndarray
_NArray = _Array | None


###########
## Class ##
###########

class R2D2(AbstractBC):##{{{
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
    
    mvq: MVQuantilesShuffle | MVRanksShuffle
    bc_method: type
    bckwargs: dict[Any]
    _bcm: AbstractBC
    _reverse: bool

    def __init__( self , col_cond: Sequence[int] = [0] , lag_search: int = 1 , lag_keep: int = 1 , bc_method: AbstractBC = CDFt , shuffle: str = "quantile" , reverse: bool = False , **kwargs: Any ) -> None:##{{{
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
        shuffle: str
            Shuffle method used, can be "quantile" or "rank".
        reverse: bool
            If False, first apply bc_method, and after the shuffle. If True, 
            reverse this operation.
        **kwargs: ...
            all others named arguments are passed to bc_method
        """
        super().__init__( "R2D2" , "NS" )
        if shuffle == "quantile":
            self.mvq = MVQuantilesShuffle( col_cond , lag_search , lag_keep )
        else:
            self.mvq = MVRanksShuffle( col_cond , lag_search , lag_keep )
        self.bc_method = bc_method
        self.bckwargs  = kwargs
        self._bcm      = None
        self._reverse  = reverse
    ##}}}
    
    @io_fit
    def fit( self , Y0: _Array  , X0: _Array  , X1: _Array  ) -> Self:##{{{
        """
        Parameters
        ----------
        Y0: numpy.ndarray | None
            Reference in calibration period
        X0: numpy.ndarray | None
            Biased model in calibration period
        X1: numpy.ndarray | None
            Biased model in projection period

        Returns
        -------
        bcm: SBCK.R2D2
            Bias Correction class fitted
        """
        self.mvq.fit(Y0)
        self._bcm = self.bc_method(**self.bckwargs)
        if self._reverse:
            Z0 = self.mvq.transform(X0)
            Z1 = self.mvq.transform(X1)
            self._bcm.fit( Y0 , Z0 , Z1 )
        else:
            self._bcm.fit( Y0 , X0 , X1 )
        
        return self
    ##}}}
    
    def _predictZ0( self , X0: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """
        Parameters
        ----------
        X0: numpy.ndarray | None
            Biased model in calibration period

        Returns
        -------
        Z0: numpy.ndarray | None
            Corrected biased model in calibration period
        """
        if X0 is None:
            return None
        if self._reverse:
            Z0 = self.mvq.transform(X0)
            Z0 = self._bcm._predictZ0( Z0 , **kwargs )
        else:
            Z0 = self._bcm._predictZ0( X0 , **kwargs )
            Z0 = self.mvq.transform(Z0)
        return Z0
    ##}}}
    
    def _predictZ1( self , X1: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """
        Parameters
        ----------
        X1: numpy.ndarray | None
            Biased model in projection period
        reinfer_X1: bool
            If the CDF of X1 must be fitted again

        Returns
        -------
        Z1: numpy.ndarray | None
            Corrected biased model in projection period
        """
        if X1 is None:
            return None
        if self._reverse:
            Z1 = self.mvq.transform(X1)
            Z1 = self._bcm._predictZ1( Z1 , **kwargs )
        else:
            Z1 = self._bcm._predictZ1( X1 , **kwargs )
            Z1 = self.mvq.transform(Z1)
        return Z1
    ##}}}
    
##}}}

@deprecated( "AR2D2 code is transfered to R2R2 since the version 2.0.0" )
class AR2D2(R2D2):##{{{
    """Deprecated, use R2D2.
    """
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:
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

