
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

from .__AbstractBC import UnivariateBC
from .__AbstractBC import MultiUBC
from .stats.__rv_extend import rv_base
from .stats.__rv_extend import rv_empirical


############
## Typing ##
############

from typing import Any
from typing import Self
from .__AbstractBC import _rv_type
from .__AbstractBC import _mrv_type
_Array = np.ndarray
_NArray = _Array | None


#############
## Classes ##
#############

class Univariate_QQD(UnivariateBC):##{{{
    """
    QQD: Quantile-Quantile of Deque (2007). The method is a simple quantile
    mapping inferred in calibration and applied in projection. Quantiles
    below and above p_left and p_right are corrected by the constants
    cst_left and cst_right, defined by:
    
    cst_left  = rvY0.icdf(p_left)  - rvX0.icdf(p_left)
    cst_right = rvY0.icdf(p_right) - rvX0.icdf(p_right)
    
    Deque, 2007: doi:10.1016/j.gloplacha.2006.11.030
    """
    
    _typeY0: type
    _typeX0: type
    _typeX1: type
    _freezeY0: bool
    _freezeX0: bool
    _freezeX1: bool
    rvY0: rv_base | None
    rvX0: rv_base | None
    rvX1: rv_base | None
    rvY1: rv_base | None
    
    def __init__( self , rvY0: _rv_type = rv_empirical , rvX0: _rv_type = rv_empirical , rvX1: _rv_type = rv_empirical , p_left: float = 0.01 , p_right: float = 0.99 ) -> None:##{{{
        """
        Parameters
        ----------
        rvY0: type | rv_base
            Law of references
        rvX0: type | rv_base
            Law of models in calibration period
        rvX1: type | rv_base
            Law of models in projection period
        p_left: float
            Minimal left quantile
        p_right: float
            Maximal right quantile
        """
        
        super().__init__( "Univariate_QQD" , "NS" )
        
        self._typeY0,self._freezeY0,self.rvY0 = self._init(rvY0)
        self._typeX0,self._freezeX0,self.rvX0 = self._init(rvX0)
        self._typeX1,self._freezeX1,self.rvX1 = self._init(rvX1)
        
        self.p_left  = p_left
        self.p_right = p_right
        self._corr_left  = 0
        self._corr_right = 0
        
    ##}}}
    
    def fit( self , Y0: _Array , X0: _Array , X1: _Array ) -> Self:##{{{
        """
        Parameters
        ----------
        Y0: numpy.ndarray
            Reference in calibration period
        X0: numpy.ndarray
            Biased model in calibration period
        X1: numpy.ndarray
            Biased model in projection period

        Returns
        -------
        bcm: SBCK.QQD
            Bias Correction class fitted
        """
        
        self.rvY0 = self._fit( Y0 , self._typeY0 , self._freezeY0 , self.rvY0 )
        self.rvX0 = self._fit( X0 , self._typeX0 , self._freezeX0 , self.rvX0 )
        self.rvX1 = self._fit( X1 , self._typeX1 , self._freezeX1 , self.rvX1 )
        
        self._corr_left  = self.rvY0.icdf(self.p_left)  - self.rvX0.icdf(self.p_left)
        self._corr_right = self.rvY0.icdf(self.p_right) - self.rvX0.icdf(self.p_right)
        
        ## First estimation of Y1
        cdfX1 = self.rvX0.cdf(X1)
        Y1    = self.rvY0.icdf(cdfX1)
        
        ## Correction of left tail
        idxL = cdfX1 < self.p_left
        if idxL.any():
            Y1[idxL] = self.rvX0.icdf(self.rvX1.cdf(X1[idxL])) + self._corr_left
        
        ## Correction of right tail
        idxR = cdfX1 > self.p_right
        if idxR.any():
            Y1[idxR] = self.rvX0.icdf(self.rvX1.cdf(X1[idxR])) + self._corr_right
        
        ## And store cdf
        self.rvY1 = rv_empirical.fit(Y1)
        
        return self
    ##}}}
    
    def _predictZ0( self , X0: _NArray , reinfer_X0: bool = False , **kwargs: Any ) -> _NArray:##{{{
        """
        Parameters
        ----------
        X0: numpy.ndarray | None
            Biased model in calibration period
        reinfer_X0: bool
            If the CDF of X0 must be fitted again

        Returns
        -------
        Z0: numpy.ndarray | None
            Corrected biased model in calibration period
        """
        
        if X0 is None:
            return None
        
        rvX0 = self.rvX0
        if reinfer_X0:
            rvX0 = self._fit( X0 , self._typeX0 , self._freezeX0 , self.rvX0 )
        Z0 = self.rvY0.icdf(rvX0.cdf(X0))
        
        return Z0
    ##}}}
    
    def _predictZ1( self , X1: _NArray , reinfer_X1: bool = False  , **kwargs: Any ) -> _NArray:##{{{
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
        
        rvX1 = self.rvX1 ## Because in QQD rvX1 dont exist!!!
        if reinfer_X1:
            rvX1 = self._fit( X1 , self._typeX1 , self._freezeX1 , self.rvX1 )
        Z1 = self.rvY1.icdf(rvX1.cdf(X1))
        
        return Z1
    ##}}}
    
##}}}

class QQD(MultiUBC):##{{{
    
    __doc__ = Univariate_QQD.__doc__

    def __init__( self ,  rvY0: _mrv_type = rv_empirical , rvX0: _mrv_type = rv_empirical , rvX1: _mrv_type = rv_empirical , p_left: float = 0.01 , p_right: float = 0.99 ):##{{{
        __doc__ = Univariate_QQD.__init__.__doc__

        ## And init upper class
        args   = tuple()
        kwargs = { 'rvY0' : rvY0 , 'rvX0' : rvX0 , 'rvX1' : rvX1 , 'p_left' : p_left , 'p_right' : p_right }
        super().__init__( "QQD" , Univariate_QQD , args = args , kwargs = kwargs )
    ##}}}
    
##}}}

