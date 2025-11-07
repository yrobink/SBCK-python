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
import scipy.stats as sc

from .__AbstractBC import AbstractBC
from .__decorators import io_fit
from .__QM   import QM
from .__QDM  import QDM

from .stats.__sparse_distance import wasserstein
from .stats.__rv_extend import rv_empirical
from .ppp.__PrePostProcessing import PrePostProcessing


############
## Typing ##
############

from typing import Self
from typing import Any
from typing import Sequence
from typing import Callable
_Array = np.ndarray
_NArray = _Array | None


###########
## Class ##
###########

class SlopeStoppingCriteria:##{{{
    """Generator to iterate until a slope is close to 0"""
    minit: int
    maxit: int
    nit: int
    tol: float
    stop: bool
    criteria: Sequence[float]
    slope: Sequence[float]
    
    def __init__( self , minit: int , maxit: int , tol: float ) -> None:
        """
        Parameters
        ----------
        minit: int
            Minimal numbers of iterations
        maxit: int
            Maximal numbers of iterations
        tol: float
            Numerical tolerance
        """
        self.minit    = minit
        self.maxit    = maxit
        self.nit      = -1
        self.tol      = tol
        self.stop     = False
        self.criteria = list()
        self.slope    = list()
    
    def initialize(self) -> None:
        self.nit      = -1
        self.stop     = False
        self.criteria = list()
        self.slope    = list()
    
    def append( self , value: float ) -> None:
        self.criteria.append(value)
        if self.nit > self.minit:
            slope,_,_,_,_ = sc.linregress( range(len(self.criteria)) , self.criteria )
            self.stop = np.abs(slope) < self.tol
            self.slope.append(slope)
    
    def __iter__(self) -> Self:
        return self
    
    def __next__(self) -> int:
        self.nit += 1
        if not self.nit < self.maxit-1:
            self.stop = True
        if not self.stop:
            return self.nit
        raise StopIteration
##}}}


class MBCn(AbstractBC):##{{{
    """MBCn Bias correction method, see [1]
    
    References
    ----------
    [1] Cannon, Alex J.: Multivariate quantile mapping bias correction: an
    N-dimensional probability density function transform for climate model
    simulations of multiple variables, Climate Dynamics, nb. 1, vol. 50, p.
    31-49, 10.1007/s00382-017-3580-6
    """
    iter_stop = SlopeStoppingCriteria
    metric = Callable
    bc = AbstractBC
    bc_params = dict[Any]
    _lbc  = Sequence[AbstractBC]
    
    def __init__( self , bc: AbstractBC = QDM , metric: Callable = wasserstein , stopping_criteria: type = SlopeStoppingCriteria , stopping_criteria_params: dict[Any] = { "minit" : 20 , "maxit" : 100 , "tol" : 1e-3 } , **kwargs: Any ) -> None: ##{{{
        """
        Parameters
        ----------
        bc  : Bias correction method
            Non stationary bias correction method, default is QDM
        metric : Callable
            Callable between two matrices, used as criteria to dermined when stopped iteration. Default is Wasserstein distance
        stopping_criteria: a class to determine when stop iteration
            See note
        stopping_criteria_params : dict
            Params of stopping_criteria
        kwargs : others named arguments
            Passed to bias correction method
        
        Note
        ----
        MBCn method used an alternance of random rotation and quantile mapping to perform the multivariate bias correction.
        At each step, the metric is used in calibration period to determine if correction is close or not of correction.
        The class SlopeStoppingCriteria compute the slope of time series of callable. When the slope is lower than tol,
        or maxit is atteigned, iterations are stopped. At least minit is performed.
        """
        super().__init__( "MBCn" , "NS" )
        self.iter_stop = stopping_criteria(**stopping_criteria_params)
        self.metric = metric
        self.bc = bc
        self.bc_params = kwargs
        self._lbc  = []
    ##}}}
    
    @property
    def maxit(self) -> int:##{{{
        return self.iter_stop.maxit
    ##}}}
    
    @property
    def nit(self) -> int:##{{{
        return self.iter_stop.nit
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
        bcm: SBCK.MBCn
            Bias Correction class fitted
        """
        
        if Y0.ndim == 1: Y0 = Y0.reshape(-1,1)
        if X0.ndim == 1: X0 = X0.reshape(-1,1)
        if X1.ndim == 1: X1 = X1.reshape(-1,1)
        
        n_features = Y0.shape[1]
        self.iter_stop.initialize()
        
        ## Generate orthogonal matrices: SO(n_features)
        self.ortho_mat  = sc.special_ortho_group.rvs( n_features , self.maxit )
        
        ## Tips for performance, inverse + ortho of next in one pass
        self.tips = np.zeros(self.ortho_mat.shape)
        for i in range(self.maxit-1):
            self.tips[i,:,:] = self.ortho_mat[i+1,:,:] @ np.linalg.inv(self.ortho_mat[i,:,:])
        self.tips[-1,:,:] = np.linalg.inv(self.ortho_mat[-1,:,:])
        
        ## Loop
        Z0_o = np.transpose( self.ortho_mat[0,:,:] @ X0.T )
        Z1_o = np.transpose( self.ortho_mat[0,:,:] @ X1.T )
        
        for i in self.iter_stop:
            Y0_o = np.transpose( self.ortho_mat[i,:,:] @ Y0.T )
            
            bc = self.bc(**self.bc_params)
            bc.fit( Y0_o , Z0_o , Z1_o )
            Z1_o,Z0_o = bc.predict(Z1_o,Z0_o)
            
            self.iter_stop.append(self.metric(Z0_o,Y0_o))
            
            self._lbc.append(bc)
            
            Z0_o = np.transpose( self.tips[i,:,:] @ Z0_o.T )
            Z1_o = np.transpose( self.tips[i,:,:] @ Z1_o.T )
        
        Z0 = np.transpose( np.linalg.inv(self.ortho_mat[self.nit,:,:]) @ Z0_o.T )
        Z1 = np.transpose( np.linalg.inv(self.ortho_mat[self.nit,:,:]) @ Z1_o.T )
        
        self.ortho_mat = self.ortho_mat[:self.nit,:,:]
        self.tips = self.tips[:self.nit,:,:]
        self.tips[-1,:,:] = np.linalg.inv(self.ortho_mat[-1,:,:])
        
        bc = self.bc(**self.bc_params)
        bc.fit( Y0 , Z0 , Z1 )
        self._lbc.append(bc)
        
        return self
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
        
        Z1_o = np.transpose( self.ortho_mat[0,:,:] @ X1.T )
        
        for i in range(self.nit):
            Z1_o = self._lbc[i]._predictZ1( Z1_o, **kwargs )
            Z1_o = np.transpose( self.tips[i,:,:] @ Z1_o.T )
        
        Z1 = self._lbc[-1]._predictZ1( Z1_o, **kwargs )
        return Z1
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
        
        Z0_o = np.transpose( self.ortho_mat[0,:,:] @ X0.T )
        
        for i in range(self.nit):
            Z0_o = self._lbc[i]._predictZ0( Z0_o , **kwargs )
            Z0_o = np.transpose( self.tips[i,:,:] @ Z0_o.T )
        
        Z0 = self._lbc[-1]._predictZ0( Z0_o , **kwargs )
        return Z0
    ##}}}
    
##}}}

class MRec(AbstractBC):##{{{
    """MRec Bias correction method, see [1]
    
    References
    ----------
    [1] Bárdossy, A. and Pegram, G.: Multiscale spatial recorrelation of RCM
    precipitation to produce unbiased climate change scenarios over large areas
    and small, Water Resources Research, 48, 9502–,
    https://doi.org/10.1029/2011WR011524, 2012.
    """
    _qmX0: QM
    _qmX1: QM
    _qmY0: QM
    _S_CY0g: _Array
    _Si_CX0g: _Array
    _re_un_mat: _Array
    n_features: int
    _rvY0: type
    _rvX0: type
    
    def __init__( self , rvY0: type = rv_empirical , rvX0: type = rv_empirical ) -> None:##{{{
        """
        Initialisation of MRec.
        
        Parameters
        ----------
        rvY0: rv_base
            Law of references
        rvX0: rv_base
            Law of models
        
        """
        super().__init__( "MRec" , "NS" )
        self._qmX0 = None
        self._qmX1 = None
        self._qmY0 = None
        self._S_CY0g  = None
        self._Si_CX0g = None
        self._re_un_mat = None
        self.n_features = 0
        self._rvY0 = rvY0
        self._rvX0 = rvX0
    ##}}}
    
    @io_fit
    def fit( self , Y0: _Array , X0: _Array , X1: _Array ) -> None:##{{{
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
        bcm: SBCK.MRec
            Bias Correction class fitted
        """
        self.n_features = Y0.shape[1]
        
        ## Transform into Gaussian data
        self._qmY0 = QM( rvY0 = sc.norm(0,1) , rvX0 = self._rvY0 ).fit( None , Y0 )
        self._qmX0 = QM( rvY0 = sc.norm(0,1) , rvX0 = self._rvX0 ).fit( None , X0 )
        self._qmX1 = QM( rvY0 = sc.norm(0,1) , rvX0 = self._rvX0 ).fit( None , X1 )
        Y0g = self._qmY0.predict(Y0)
        X0g = self._qmX0.predict(X0)
        
        ## Correlation matrix
        CY0g = np.corrcoef( Y0g.T )
        CX0g = np.corrcoef( X0g.T )
        
        ## Squareroot matrix
        a_CY0g,d_CY0g,_ = np.linalg.svd(CY0g)
        self._S_CY0g = a_CY0g @ np.diag(np.sqrt(d_CY0g)) @ a_CY0g.T
        
        a_CX0g,d_CX0g,_ = np.linalg.svd(CX0g)
        self._Si_CX0g = a_CX0g @ np.diag( np.power(d_CX0g,-0.5) ) @ a_CX0g.T
        
        ## Decor-recor-relation
        self._re_un_mat = self._S_CY0g @ self._Si_CX0g
        X0_recor = np.transpose( self._re_un_mat @ X0g.T )
        
        ## Final QM
        self._qmY0 = QM( rvY0 = self._rvY0 , rvX0 = sc.norm ).fit( Y0 , X0_recor )
        
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
        X0g = self._qmX0.predict( X0 , **kwargs )
        X0r = np.transpose( self._re_un_mat @ X0g.T )
        Z0  = self._qmY0.predict(X0r)
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
        X1g = self._qmX1.predict( X1 , **okwargs )
        X1r = np.transpose( self._re_un_mat @ X1g.T )
        Z1  = self._qmY0.predict(X1r)
        return Z1
    ##}}}
    
##}}}


class XClimSPPP(PrePostProcessing):##{{{
    """
    Experimental: just a class based on SBCK.ppp.PrePostProcessing for xclim,
    stationary case
    """
    def __init__( self , **kwargs: Any ) -> None:
        """
        kwargs are directly given to SBCK.ppp.PrePostProcessing, only keywords
        arguments are available.
        
        """
        PrePostProcessing.__init__( self , **kwargs )
    
    def fit( self , Y0: _NArray , X0: _NArray , X1: _NArray = None ) -> Self:
        """
        Parameters
        ----------
        Y0: numpy.ndarray | None
            Reference in calibration period
        X0: numpy.ndarray | None
            Biased model in calibration period
        X1: numpy.ndarray | None
            Not used

        Returns
        -------
        bcm: SBCK.XClimSPPP
            Bias Correction class fitted
        """
        PrePostProcessing.fit( self , Y0 = Y0 , X0 = X0 )
        
        return self
    
    def predict( self , X1: _NArray , X0: _NArray = None ) -> _NArray:
        """
        Parameters
        ----------
        X1: numpy.ndarray | None
            Biased model in CALIBRATION period
        X0: numpy.ndarray | None
            Not used

        Returns
        -------
        Z1: numpy.ndarray | None
            Correction of biased model in CALIBRATION period
        """
        return PrePostProcessing.predict( self , X1 )
##}}}

class XClimNPPP(PrePostProcessing):##{{{
    """
    Experimental: just a class based on SBCK.ppp.PrePostProcessing for xclim,
    non-stationary case
    """
    def __init__( self , **kwargs: Any ) -> None:
        """
        kwargs are directly given to SBCK.ppp.PrePostProcessing, only keywords
        arguments are available.
        """
        PrePostProcessing.__init__( self , **kwargs )
    
    def fit( self , Y0: _NArray , X0: _NArray , X1: _NArray) -> Self:
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
        bcm: SBCK.XClimNPPP
            Bias Correction class fitted
        """
        PrePostProcessing.fit( self , Y0 = Y0 , X0 = X0 , X1 = X1 )
        
        return self
    
    def predict( self , X1:_NArray , X0: _NArray = None ) -> _NArray | tuple[_NArray,_NArray]:
        """
        Parameters
        ----------
        X1: numpy.ndarray | None
            Biased model in projection period
        X0: numpy.ndarray | None
            Biased model in calibration period

        Returns
        -------
        Z1: numpy.ndarray | None
            Correction of biased model in projection period
        Z0: numpy.ndarray | None
            Correction of biased model in calibration period
        """
        return PrePostProcessing.predict( self , X1 = X1 , X0 = X0 )
##}}}

