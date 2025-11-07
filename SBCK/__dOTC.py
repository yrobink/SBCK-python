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

import numpy       as np

from .__AbstractBC import AbstractBC
from .__AbstractBC import UnivariateBC
from .__AbstractBC import MultiUBC
from .__QM         import QM
from .__decorators import io_fit

from .stats.__SparseHist import SparseHist
from .stats.__SparseHist import bin_width_estimator
from .stats.__rv_extend import rv_empirical
from .stats.__transport import POTemd
from .misc.__linalg import sqrtm

############
## Typing ##
############

from typing import Self
from typing import Any
from .__AbstractBC import _rv_type
from .__AbstractBC import _mrv_type
_Array = np.ndarray
_NArray = _Array | None


###########
## Class ##
###########

class OTC(AbstractBC):##{{{
    """Optimal Transport bias Corrector, see [1]
    
    Note: Only the center of the bins associated to the corrected points are
    returned, but all corrections of the form:
    >> otc.predict(X0) + np.random.uniform( low = - otc.bin_width / 2 , high = otc.bin_width / 2 , size = X0.shape[0] )
    are equivalent for OTC.
    
    Attributes
    ----------
    muY    : SBCK.SparseHist
        Multivariate histogram of references
    muX    : SBCK.SparseHist
        Multivariate histogram of biased dataset
    
    References
    ----------
    [1] Robin, Y., Vrac, M., Naveau, P., Yiou, P.: Multivariate stochastic bias
    corrections with optimal transport, Hydrol. Earth Syst. Sci., 23, 773–786,
    2019, https://doi.org/10.5194/hess-23-773-2019
    """
    
    muX: SparseHist
    muY: SparseHist
    bin_width: _Array
    bin_origin: _Array
    _plan: _Array
    _ot: POTemd
    
    def __init__( self , bin_width: _NArray = None , bin_origin: _NArray = None , ot: POTemd = POTemd() ) -> None:##{{{
        """
        Parameters
        ----------
        bin_width  : numpy.ndarray | None
            Lenght of bins, see SBCK.stats.SparseHist. If is None, it is estimated during the fit
        bin_origin : numpy.ndarray | None
            Corner of one bin, see SBCK.stats.SparseHist. If is None, 0 is used
        ot         : OT*Solver*
            A solver for Optimal transport, default is POTemd()
        
        """
        
        super().__init__( "OTC" , "S" )
        self.muX = None
        self.muY = None
        self.bin_width  = bin_width
        self.bin_origin = bin_origin
        self._plan       = None
        self._ot         = ot
    ##}}}
    
    @io_fit
    def fit( self , Y0: _Array , X0: _Array ) -> Self:##{{{
        """
        Parameters
        ----------
        Y0: numpy.ndarray | None
            Reference in calibration period
        X0: numpy.ndarray | None
            Biased model in calibration period
        """
        
        ## Sparse Histogram
        self.bin_width  = np.array( [self.bin_width ] ).ravel() if self.bin_width  is not None else bin_width_estimator( Y0 , X0 )
        self.bin_origin = np.array( [self.bin_origin] ).ravel() if self.bin_origin is not None else np.zeros( self.bin_width.size )
        
        self.bin_width  = np.array( [self.bin_width] ).ravel()
        self.bin_origin = np.array( [self.bin_origin] ).ravel()
        
        self.muY = SparseHist( Y0 , bin_width = self.bin_width , bin_origin = self.bin_origin )
        self.muX = SparseHist( X0 , bin_width = self.bin_width , bin_origin = self.bin_origin )
        
        
        ## Optimal Transport
        self._ot.fit( self.muX , self.muY )
        
        ## 
        self._plan = np.copy( self._ot.plan() )
        self._plan = ( self._plan.T / self._plan.sum( axis = 1 ) ).T
        
        return self
    ##}}}
    
    def _predictZ0( self , X0: _Array , reinfer_X0: bool = False , **kwargs: Any ) -> _Array:##{{{
        """
        Parameters
        ----------
        X0: numpy.ndarray
            Biased model in calibration period
        reinfer_X0: bool
            If the law of X0 must be fitted again

        Returns
        -------
        Z0: numpy.ndarray | None
            Corrected biased model in calibration period
        """
        
        if reinfer_X0:
            muX = SparseHist( X0 , bin_width = self.bin_width , bin_origin = self.bin_origin )
            ot  = type(self._ot)()
            ot.fit( muX , self.muY )
            plan = np.copy( ot.plan() )
            plan = ( plan.T / plan.sum( axis = 1) ).T
        else:
            muX = self.muX
            plan = self._plan
    
        indx = muX.argwhere(X0)
        indy = np.zeros_like(indx)
        for i,ix in enumerate(indx):
            indy[i] = np.random.choice( range(self.muY.sizep) , p = plan[ix,:] )
        Z0 = self.muY.c[indy,:]
        
        return Z0
    ##}}}
    
##}}}

class dOTC(AbstractBC):##{{{
    """
    Description
    -----------
    Dynamical Optimal Transport bias Corrector, taking account of an evolution of the distribution. see [1]
    
    Note: Only the center of the bins associated to the corrected points are
    returned, but all corrections of the form:
    >> dotc.predict(X1) + np.random.uniform( low = - dotc.bin_width / 2 , high = dotc.bin_width / 2 , size = X1.shape[0] )
    are equivalent for dOTC.

    Attributes
    ----------
    otc   : SBCK.OTC
        OTC corrector between X1 and the estimation of Y1
    
    References
    ----------
    [1] Robin, Y., Vrac, M., Naveau, P., Yiou, P.: Multivariate stochastic bias
    corrections with optimal transport, Hydrol. Earth Syst. Sci., 23, 773–786,
    2019, https://doi.org/10.5194/hess-23-773-2019
    """
    
    _cov_factor_str: str
    _cov_factor: _Array | None
    _otcX0Y0: OTC
    otc: OTC
    _ot: POTemd

    def __init__( self , bin_width: _NArray = None , bin_origin: _NArray = None , cov_factor: str = "std" , ot: POTemd = POTemd() ) -> None:##{{{
        """
        Parameters
        ----------
        bin_width  : numpy.ndarray | None
            Lenght of bins, see SBCK.stats.SparseHist. If is None, it is estimated during the fit
        bin_origin : numpy.ndarray | None
            Corner of one bin, see SBCK.stats.SparseHist. If is None, 0 is used
        cov_factor : str
            Correction factor during transfer of the evolution between X0 and X1 to Y0
                "cholesky" => compute the cholesky factor
                "sqrtm"    => compute the square root matrix factor
                "std"      => compute the standard deviation factor
                "id"       => identity is used
        ot         : OT*Solver*
            A solver for Optimal transport, default is POTemd()
        
        """
        super().__init__("dOTC","NS")
        self._cov_factor = None
        if type(cov_factor) is str:
            if cov_factor not in ["cholesky","sqrtm","std","id"]:
                raise ValueError("'cov_factor' must be 'cholesky', 'sqrtm', 'std' or 'id'")
            self._cov_factor_str = cov_factor
        else:
            try:
                self._cov_factor = np.array([cov_factor])
            except Exception:
                raise ValueError("cov_factor not a string and not castable to numpy array")
        
        self.bin_width  = bin_width
        self.bin_origin = bin_origin
        self._otcX0Y0   = None
        self.otc        = None
        self._ot        = ot
    ##}}}
    
    @io_fit
    def fit( self , Y0: _Array , X0: _Array , X1: _Array ) -> Self:##{{{
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
        bcm: SBCK.dOTC
            Bias Correction class fitted
        """
        ## Set the covariance factor correction
        
        if Y0.ndim == 1: Y0 = Y0.reshape(-1,1)
        if X0.ndim == 1: X0 = X0.reshape(-1,1)
        if X1.ndim == 1: X1 = X1.reshape(-1,1)
        
        if self._cov_factor is None:
            if self._cov_factor_str in ["std","sqrtm","cholesky"]:
                if Y0.shape[1] == 1:
                    self._cov_factor = np.std( Y0 ) / np.std( X0 )
                    self._cov_factor = np.array([self._cov_factor]).reshape(1,1)
                elif self._cov_factor_str == "cholesky":
                    fact0 = np.linalg.cholesky( np.cov( Y0 , rowvar = False ) )
                    fact1 = np.linalg.cholesky( np.cov( X0 , rowvar = False ) )
                    self._cov_factor = np.dot( fact0 , np.linalg.inv( fact1 ) )
                elif self._cov_factor_str == "sqrtm":
                    fact0 = sqrtm( np.cov( Y0 , rowvar = False ) )
                    fact1 = sqrtm( np.cov( X0 , rowvar = False ) )
                    self._cov_factor = np.dot( fact0 , np.linalg.inv( fact1 ) )
                else:
                    fact0 = np.std( Y0 , axis = 0 )
                    fact1 = np.std( X0 , axis = 0 )
                    self._cov_factor = np.diag( fact0 / fact1 )
            else:
                self._cov_factor = np.identity(Y0.shape[1])
        self._cov_factor = self._cov_factor.reshape(self.ndim,self.ndim)
        self.bin_width = self.bin_width if self.bin_width is not None else bin_width_estimator( Y0 , X0 , X1 )
        
        
        ## Optimal plan
        otcY0X0 = OTC( self.bin_width , self.bin_origin , ot = self._ot )
        otcX0X1 = OTC( self.bin_width , self.bin_origin , ot = self._ot )
        otcY0X0.fit( X0 , Y0 )
        otcX0X1.fit( X1 , X0 )
        self._otcX0Y0 = OTC( self.bin_width , self.bin_origin , ot = self._ot )
        self._otcX0Y0.fit(Y0,X0)
        
        ## Estimation of Y1
        yX0 = otcY0X0.predict(Y0)
        yX1 = otcX0X1.predict(yX0)
        motion = yX1 - yX0
        Y1 = Y0 + (self._cov_factor @ motion.T).T
        
        ## Optimal plan for correction
        self.otc = OTC( self.bin_width , self.bin_origin , ot = self._ot )
        self.otc.fit( Y1 , X1 )
        
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
        Z0 = self._otcX0Y0.predict( X0 , **kwargs )
        return Z0
    ##}}}
    
    def _predictZ1( self , X1: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """
        Parameters
        ----------
        X1: numpy.ndarray | None
            Biased model in projection period

        Returns
        -------
        Z1: numpy.ndarray | None
            Corrected biased model in projection period
        """
        if X1 is None:
            return None
        okwargs = dict(kwargs)
        if okwargs.get("reinfer_X1",False):
            okwargs["reinfer_X0"] = True
        Z1 = self.otc.predict( X1 , **okwargs )
        return Z1
    ##}}}
    
##}}}


class Univariate_dOTC1d(UnivariateBC):##{{{
    """One dimensionnal version of dOTC, use quantile mapping (instead of simplex)
    to solve the transport problem (very very very faster).
    
    References
    ----------
    [1] Robin, Y., Vrac, M., Naveau, P., Yiou, P.: Multivariate stochastic bias
    corrections with optimal transport, Hydrol. Earth Syst. Sci., 23, 773–786,
    2019, https://doi.org/10.5194/hess-23-773-2019
    """
    
    _rvY0: _rv_type
    _rvX0: _rv_type
    _rvX1: _rv_type
    _planX0Y0: QM | None
    _planX1Y1: QM | None
    _cfactor: float | None

    def __init__( self , rvY0: _rv_type = rv_empirical , rvX0: _rv_type = rv_empirical , rvX1: _rv_type = rv_empirical , **kwargs: Any ) -> None:##{{{
        """
        Parameters
        ----------
        rvY0: type | rv_base
            Law of references
        rvX0: type | rv_base
            Law of models in calibration period
        rvX1: type | rv_base
            Law of models in projection period
        """
        
        super().__init__( "dOTC1d" , "NS" )
        
        self._rvY0 = rvY0
        self._rvX0 = rvX0
        self._rvX1 = rvX1
        self._planX0Y0 = None
        self._planX1Y1 = None
        
        cfactor = kwargs.get("cfactor")
        if cfactor is not None:
            cfactor = float(cfactor)
        self._cfactor = cfactor
        
    ##}}}
    
    def fit( self , Y0: _Array , X0: _Array , X1: _Array ) -> Self:##{{{
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
        bcm: SBCK.dOTC1d
            Bias Correction class fitted
        """
        
        ## cfactor
        if self._cfactor is None:
            self._cfactor  = Y0.std() / X0.std()
        
        ## Inference of Y1
        D0  = QM( rvY0 = self._rvX0 , rvX0 = self._rvY0 ).fit( X0 , Y0 ).predict(Y0)
        D1  = QM( rvY0 = self._rvX1 , rvX0 = self._rvX0 ).fit( X1 , X0 ).predict(D0)
        D10 = self._cfactor * (D1 - D0)
        Y1  = Y0 + D10
        
        ##
        self._planX0Y0 = QM( rvY0 = self._rvY0 , rvX0 = self._rvX0 ).fit( Y0 , X0 )
        self._planX1Y1 = QM(                     rvX0 = self._rvX1 ).fit( Y1 , X1 )
        
        return self
    ##}}}
    
    def _predictZ1( self , X1: _NArray , reinfer_X1: bool = False , **kwargs: Any ) -> _NArray:##{{{
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
        if reinfer_X1:
            Z1 = QM( rvY0 = self._planX1Y1._rvY0 ).fit(None,X1).predict(X1)
        else:
            Z1 = self._planX1Y1.predict(X1)
        return Z1
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
        if reinfer_X0:
            Z0 = QM( rvY0 = self._planX0Y0._rvY0 ).fit(None,X0).predict(X0)
        else:
            Z0 = self._planX0Y0.predict(X0)
        return Z0
    ##}}}
    
##}}}

class dOTC1d(MultiUBC):##{{{
    __doc__ = Univariate_dOTC1d.__doc__
    
    def __init__( self , rvY0: _mrv_type = rv_empirical , rvX0: _mrv_type = rv_empirical , rvX1: _mrv_type = rv_empirical , **kwargs: Any ) -> None:
        __doc__ = Univariate_dOTC1d.__init__.__doc__
        args    = tuple()
        gkwargs = { **kwargs , **{ 'rvY0' : rvY0 , 'rvX0' : rvX0 , 'rvX1' : rvX1 } }
        super().__init__( "dOTC1d" , Univariate_dOTC1d , args = args , kwargs = gkwargs )
##}}}

