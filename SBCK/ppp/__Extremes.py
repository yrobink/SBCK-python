
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

import numpy as np
from .__PrePostProcessing import PrePostProcessing
from ..misc.__sys import deprecated


############
## Typing ##
############

from typing import Sequence
from typing import Any

_Array = np.ndarray
_Cols = Sequence[int] | int | None


###########
## Class ##
###########

class LimitTailsRatio(PrePostProcessing):##{{{
    """This class is used to post-process the tails of the correction, in the
    case where too large values are produced.
    
    The idea (for the right tail) is to compare the max of the correction with
    the max defined by
        min( ratio , max of obs * (Q95 - Q50 projection) / (Q95 - Q50) calibration ). 
    If the max is greater than this estimation, the tail beyond the quantile 95%
    is rescaled to the interval [Q95%,estimated max].
    
    """
    
    _cols: _Cols
    _ratio = float
    _tails = float
    
    _p_r   = float
    _p_l   = float
    _p_c   = float
    
    _qcY0  = float | None
    _maxY0 = float | None
    _minY0 = float | None
    _qrX0  = float | None
    _qcX0  = float | None
    _qlX0  = float | None
    _qrX1  = float | None
    _qcX1  = float | None
    _qlX1  = float | None


    def __init__( self , *args: Any , ratio: float = 1.5 , p_r: float = 0.95 , p_l: float = 0.05 , p_c: float = 0.5 , tails: str = "both" , cols: _Cols = None , **kwargs: Any ) -> None:##{{{
        """
        Arguments
        ---------
        ratio: float
            Maximal ratio to increase observed extreme
        p_r: float
            Right quantile to estimated the right tail. 95% in the example.
        p_l: float
            Left quantile to estimate the left tails. 5% in the example.
        p_c: float
            Center quantile to estimated the tails. 50% in the example.
        tails: str
            Tails to apply the PPP. Can be "left", "right" or "both".
        cols: int or array of int
            The columns to apply
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        *kwargs:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        """
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name = "LimitTailsRatio"
        
        self._cols = cols
        if cols is not None:
            self._cols = np.array( [cols] , dtype = int ).squeeze().reshape(-1)
        
        self._ratio = ratio
        self._tails = tails
        
        self._p_r   = p_r
        self._p_l   = p_l
        self._p_c   = p_c
        
        self._qcY0  = None
        self._maxY0 = None
        self._minY0 = None
        
        self._qrX0  = None
        self._qcX0  = None
        self._qlX0  = None
        
        self._qrX1  = None
        self._qcX1  = None
        self._qlX1  = None
        
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """transform"""
        
        if self._kind == "Y0":
            self._qcY0  = np.quantile( X , self._p_c , axis = 0 )
            self._maxY0 = np.max( X , axis = 0 )
            self._minY0 = np.min( X , axis = 0 )
        if self._kind == "X0":
            self._qrX0 = np.quantile( X , self._p_r , axis = 0 )
            self._qcX0 = np.quantile( X , self._p_c , axis = 0 )
            self._qlX0 = np.quantile( X , self._p_l , axis = 0 )
        if self._kind == "X1":
            self._qrX1 = np.quantile( X , self._p_r , axis = 0 )
            self._qcX1 = np.quantile( X , self._p_c , axis = 0 )
            self._qlX1 = np.quantile( X , self._p_l , axis = 0 )
        
        return X
    ##}}}
    
    def itransform( self , Xt: _Array ) -> _Array:##{{{
        """inverse transform"""
        X  = Xt.copy()
        if self._kind == "X1":
            
            ## Identify cols
            cols = self._cols
            if cols is None:
                cols = np.array( [np.arange( 0 , Xt.shape[1] , dtype = int )] ).squeeze()
            
            ## Right tail
            if self._tails in ["right","both"]:
                S  = (self._qrX1 - self._qcX1) / (self._qrX0 - self._qcX0)
                S  = np.where( S > self._ratio , self._ratio , S ) * ( self._maxY0 - self._qcY0 ) + self._qcY0 + (self._qcX1 - self._qcX0)
                M  = Xt.max( axis = 0 )
                Q  = np.quantile( Xt , self._p_r , axis = 0 )
                Xt = np.where( (M < S) | (Xt < Q) , Xt , (Xt - Q) / (M - Q) * (S - Q) + Q )
                X[:,cols] = Xt[:,cols]
            
            ## Left tail
            if self._tails in ["left","both"]:
                S  = (self._qcX1 - self._qlX1) / (self._qcX0 - self._qlX0)
                S  = np.where( S > self._ratio , self._ratio , S ) * (self._qcY0 - self._minY0) + self._qcY0 + (self._qcX1 - self._qcX0)
                M  = Xt.min( axis = 0 )
                Q  = np.quantile( Xt , self._p_l , axis = 0 )
                Xt = np.where( (M > S) | (Xt > Q) , Xt , (Xt - Q) / (M - Q) * (S - Q) + Q )
                X[:,cols] = Xt[:,cols]
            
        return X
    ##}}}
    
##}}}


################
## Deprecated ##
################

@deprecated( "PPPLimitTailsRatio is renamed LimitTailsRatio since the version 2.0.0" )
class PPPLimitTailsRatio(LimitTailsRatio):##{{{
    
    """See SBCK.ppp.LimitTailsRatio"""

    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPLimitTailsRatio"
    ##}}}
    
##}}}

