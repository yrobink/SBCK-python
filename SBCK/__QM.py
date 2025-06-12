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
import scipy.stats as sc

from .__AbstractBC import UnivariateBC
from .__AbstractBC import MultiUBC
from .stats.__rv_extend import rv_base
from .stats.__rv_extend import rv_empirical


############
## Typing ##
############

from typing import Self
from typing import Any
from typing import Sequence
from .__AbstractBC import _rv_type
from .__AbstractBC import _mrv_type
_Array = np.ndarray
_NArray = _Array | None


###########
## Class ##
###########

class Univariate_QM(UnivariateBC):##{{{
    """Quantile Mapping bias corrector, see e.g. [1,2]. The implementation
    proposed here is generic, and can use scipy.stats to fit a parametric
    distribution, or can use a frozen distribution.
    
    Example
    -------
    ```
    ## Start with imports
    from SBCK import QM
    from SBCK.stats import rv_empirical
    
    ## Start by define two kinds of laws from scipy.stats, the Normal and Exponential distribution
    norm  = sc.norm
    expon = sc.expon
    
    ## And define calibration and projection dataset such that the law of each columns are reversed
    size = 10000
    Y0   = np.stack( [expon.rvs( scale = 2 , size = size ),norm.rvs( loc = 0 , scale = 1 , size = size )] ).T
    X0   = np.stack( [norm.rvs( loc = 0 , scale = 1 , size = size ),expon.rvs( scale = 1 , size = size )] ).T
    
    ## Generally, the law of Y0 and X0 is unknow, so we use the empirical histogram distribution
    qm   = bc.QM( rvY0 = [rv_empirical,rv_empirical] , rvX0 = [rv_empirical,rv_empirical] ).fit( Y0 , X0 )
    Z0_h = qm.predict(X0)
    
    ## Actually, this is the default behavior
    qm = bc.QM().fit( Y0 , X0 )
    assert np.abs(Z0_h - qm.predict(X0)).max() < 1e-12
    
    ## In some case we know the kind of law of Y0 (or X0)
    qm    = bc.QM( rvY0 = [expon,norm] ).fit( Y0 , X0 )
    Z0_Y0 = qm.predict(X0)
    
    ## Or, even better, we know the law of the 2nd component of Y0 (or X0)
    qm    = bc.QM( rvY0 = [expon,norm(loc=0,scale=1)] ).fit( Y0 , X0 )
    Z0_Y2 = qm.predict(X0)
    
    ## Obviously, we can mix all this strategy to build a custom Quantile Mapping
    qm    = bc.QM( rvY0 = [rv_empirical,norm(loc=0,scale=1)] , rvX0 = [norm,rv_empirical] ).fit( Y0 , X0 )
    Z0_Yh = qm.predict(X0)
    ```
    
    References
    ----------
    [1] Panofsky, H. A. and Brier, G. W.: Some applications of statistics to
    meteorology, Mineral Industries Extension Services, College of Mineral
    Industries, Pennsylvania State University, 103 pp., 1958.
    [2] Wood, A. W., Leung, L. R., Sridhar, V., and Lettenmaier, D. P.:
    Hydrologic Implications of Dynamical and Statistical Approaches to
    Downscaling Climate Model Outputs, Clim. Change, 62, 189–216,
    https://doi.org/10.1023/B:CLIM.0000013685.99609.9e, 2004.
    """
    
    _typeX0: type
    _typeY0: type
    _freezeX0: bool
    _freezeY0: bool
    rvY0: rv_base | None
    rvX0: rv_base | None
    
    def __init__( self , rvY0: _rv_type = rv_empirical , rvX0: _rv_type = rv_empirical ) -> None:##{{{
        """
        Parameters
        ----------
        rvY0: type | rv_base
            Law of references
        rvX0: type | rv_base
            Law of models
        """
        super().__init__( "Univariate_QM" , "S" )
        self._typeY0,self._freezeY0,self.rvY0 = self._init(rvY0)
        self._typeX0,self._freezeX0,self.rvX0 = self._init(rvX0)
    ##}}}
    
    def fit( self , Y0: _NArray , X0: _NArray ) -> Self:##{{{
        """
        Parameters
        ----------
        Y0: numpy.ndarray | None
            Reference in calibration period
        X0: numpy.ndarray | None
            Biased model in calibration period

        Returns
        -------
        bcm: SBCK.QM
            Bias Correction class fitted
        """
        
        self.rvY0 = self._fit( Y0 , self._typeY0 , self._freezeY0 , self.rvY0 )
        self.rvX0 = self._fit( X0 , self._typeX0 , self._freezeX0 , self.rvX0 )
        
        return self
    ##}}}
    
    def _predictZ0( self , X0: _NArray , reinfer_X0: bool = False , **kwargs ) -> _NArray:##{{{
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
        cdfX0 = self.rvX0.cdf
        if reinfer_X0:
            rvX0 = self._fit( X0 , self._typeX0 , self._freezeX0 , self.rvX0 )
            cdfX0 = rvX0.cdf
        eps  = np.sqrt(np.finfo(X0.dtype).resolution)
        cdf  = cdfX0(X0)
        cdfx = max( 1 - ( 1 - cdf[cdf < 1].max() / 10 ) , 1 - eps )
        cdfn = min(           cdf[cdf > 0].min() / 10   ,     eps )
        cdf = np.where( cdf < 1 , cdf , cdfx )
        cdf = np.where( cdf > 0 , cdf , cdfn )
        
        return self.rvY0.icdf(cdf)
    ##}}}
    
##}}}

class QM(MultiUBC):##{{{
    __doc__ = Univariate_QM.__doc__
    
    def __init__( self , rvY0: _mrv_type = rv_empirical , rvX0: _mrv_type = rv_empirical , **kwargs ) -> None:##{{{
        __doc__ = Univariate_QM.__init__.__doc__
        
        ## Init upper class
        args   = tuple()
        kwargs = { **{ 'rvY0' : rvY0 , 'rvX0' : rvX0 } , **kwargs }
        super().__init__( "QM" , Univariate_QM , args = args , kwargs = kwargs )
    ##}}}
    
##}}}

