
## Copyright(c) 2024, 2025 Yoann Robin
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

from ..misc.__linalg import sqrtm


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

class MNPar:##{{{
    """Class to manage (multivariate) gaussian transformations

    Properties
    ----------
    ndim: int
        Numbers of dimensions
    univariate: bool
        If normalization applied must be univariate
    m: numpy.ndarray
        The mean
    s: numpy.ndarray
        The standard deviation
    C: numpy.ndarray
        The covariance matrix
    S: numpy.ndarray
        The matrix square root of the covariance matrix
    ivs: numpy.ndarray
        Inverse of standard deviation
    ivS: numpy.ndarray
        Inverse of the matrix square root of the covariance matrix
    """
    
    ndim: int
    univariate: bool
    _m: _Array
    _s: _Array
    _C: _Array
    _S: _Array
    _ivs: _Array
    _ivS: _Array

    def __init__( self , X: _Array , univariate: bool = False ) -> None:##{{{
        """
        Arguments
        ---------
        X: numpy.ndarray
            A multivariate dataset
        univariate: bool
            If true, covariance matrix is replaced by standard deviation
        """
        if X.ndim == 1:
            X = X.reshape(-1,1)
        self.ndim = X.shape[1]
        self.univariate = univariate
        self._m   = X.mean(0).reshape(1,self.ndim)
        self._s   = X.std(0).reshape(1,self.ndim)
        self._C   = np.cov( X , rowvar = False ).reshape(self.ndim,self.ndim)
        self._S   = sqrtm(self.C)
        self._ivs = 1. / self.s
        self._ivS = np.linalg.pinv(self.S)
    ##}}}
    
    ## Normalization methods ##{{{
    
    def _unormalize( self , X: _Array ) -> _Array:
        """Center scale, assuming data is univariate"""
        return ( self.ivs * (X - self.m) )

    def _mnormalize( self , X: _Array ) -> _Array:
        """Center scale, assuming data is multivariate"""
        return ( self.ivS @ (X - self.m).T ).T

    def normalize( self , X: _Array ) -> _Array:
        """Center scale data"""
        if self.univariate or self.ndim == 1:
            N = self._unormalize(X.reshape(-1,self.ndim))
        else:
            N = self._mnormalize(X.reshape(-1,self.ndim))
        return N.reshape(-1,self.ndim)
    
    def _iunormalize( self , X: _Array ) -> _Array:
        """Inverse center scale, assuming data is univariate"""
        return self.s * X + self.m

    def _imnormalize( self , X: _Array ) -> _Array:
        """Inverse center scale, assuming data is multivariate"""
        return ( self.S @ X.T ).T + self.m

    def inormalize( self , N: _Array ) -> _Array:
        """Inverse center scale data"""
        if self.univariate or self.ndim == 1:
            X = self._iunormalize(N.reshape(-1,self.ndim))
        else:
            X = self._imnormalize(N.reshape(-1,self.ndim))
        return X.reshape(-1,self.ndim)
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def m(self) -> _Array:
        return self._m
    
    @property
    def s(self) -> _Array:
        return self._s
    
    @property
    def C(self) -> _Array:
        return self._C
    
    @property
    def S(self) -> _Array:
        return self._S
    
    @property
    def ivs(self) -> _Array:
        return self._ivs
    
    @property
    def ivS(self) -> _Array:
        return self._ivS
    
    ##}}}
    
##}}}

class UMNAdjust(PrePostProcessing):##{{{
    """Pre Post Processing for Gaussian normalization"""
    
    univariate: bool
    _p: dict[str,MNPar]

    def __init__( self , *args: Any , univariate: bool = False , **kwargs: Any ):##{{{
        """
        Arguments
        ---------
        univariate : bool
            Assume or not that the marginals are independent
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        *kwargs:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        """
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name      = "UMNAdjust"
        self.univariate = univariate
        self._p         = {}
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """Apply transformation"""
        self._p[self._kind] = MNPar( X , univariate = self.univariate )
        NX  = self._p[self._kind].normalize(X)
        return NX
    ##}}}
    
    def itransform( self , Xt: _Array  ) -> _Array:##{{{
        """Apply inverse transformation"""
        
        pXt = MNPar( Xt , univariate = self.univariate )
        NXt = pXt.normalize(Xt)
        
        match self._kind:
            case 'X0':
                X = self._p['Y0'].inormalize(Xt)
            case 'X1':
                if self.univariate:
                    X  =   self._p['X1'].s * self._p['X0'].ivs * self._p['Y0'].s * NXt
                else:
                    X  = ( self._p['X1'].S @ self._p['X0'].ivS @ self._p['Y0'].S @ NXt.T ).T
                X  = X + self._p['Y0'].s * self._p['X0'].ivs * (self._p['X1'].m - self._p['X0'].m)
                X  = X + self._p['X1'].s * self._p['X0'].ivs * self._p['Y0'].s * pXt.ivs * pXt.m
                X  = X + self._p['Y0'].m
        
        return X
    ##}}}
    
##}}}

