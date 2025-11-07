
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
import scipy.interpolate as sci
import scipy.optimize as sco

from ..__AbstractBC import UnivariateBC
from ..__AbstractBC import MultiUBC

from ..stats.__rv_extend import rv_base
from ..stats.__rv_extend import rv_empirical

from ..__QM import QM


############
## Typing ##
############

from typing import Sequence
from typing import Callable
from typing import Self

_Array = np.ndarray


#############
## Classes ##
#############

class Univariate_AlphaPooling(UnivariateBC):##{{{
    
    _typeY0: type
    _typeX0: type
    _typeX1: type
    rvY0: rv_base | None
    rvX0: list[rv_base | None]
    rvX1: list[rv_base | None]
    _fa: bool = False
    _a: float | None
    _igpool: Callable
    _w: _Array | None = None
    _qm0: QM | None = None
    _qm1: QM | None = None
    
    def __init__( self , alpha: float | None = None , rvY0: rv_base = rv_empirical , rvX0: rv_base = rv_empirical , rvX1: rv_base = rv_empirical ) -> None:##{{{
        
        super().__init__( "Univariate_AlphaPooling" , "NS" )
        
        self._typeY0 = rvY0
        self._typeX0 = rvX0
        self._typeX1 = rvX1
        
        self._a      = alpha
        if isinstance(alpha,float):
            self._fa = True
        
    ##}}}
    
    def _normalization( self , Y0: _Array , X0s: Sequence[_Array] , X1s: Sequence[_Array] ) -> tuple[Sequence[_Array],Sequence[_Array]]:##{{{
        """Normalization to preprocess data
        Center the mean of X0s to the mean of Y0 (and shift X1s accordingly)
        """
        mX1s = np.mean([X1.mean() for X1 in X1s])
        mX0s = np.mean([X0.mean() for X0 in X0s])
        mY0  = Y0.mean()
        NX0s = [ X0 - mX0s + mY0 for X0 in X0s ]
        NX1s = [ X1 - mX0s + mY0 for X1 in X1s ]

        return NX1s,NX0s
    ##}}}
    
    def _gpool( self , x: _Array ) -> _Array:##{{{
        """G-pooling function
        
        Defined by:
        y = ( x^alpha - (1-x)^alpha ) / alpha

        """
        return ( x**self.a - (1-x)**self.a ) / self.a
    ##}}}
    
    def _pooling( self , rvXs: Sequence[rv_base] ) -> rv_empirical:##{{{
        """Apply pooling operator to the sequence of random variables"""
        
        ## Find support
        xmin =  1e9
        xmax = -1e9
        for rvX in rvXs:
            xmin = min( xmin , rvX.icdf(1e-6) )
            xmax = max( xmax , rvX.icdf(1-1e-6) )
        x   = np.linspace( xmin , xmax , 1000 )
        
        ## And find the combination
        zg   = np.array( [ self._w[i] * (rvX.cdf(x)**self._a - (1-rvX.cdf(x))**self._a) for i,rvX in enumerate(rvXs) ] ).sum(0) / self._a
        y    = self._igpool(zg)
        cdf  = sci.interp1d( x , y , bounds_error = False , fill_value = (0,1) )
        icdf = sci.interp1d( y , x , bounds_error = False , fill_value = (xmin,xmax) )
        rvG  = rv_empirical( cdf , icdf , None )
        
        return rvG
    ##}}}
    
    def _fit_pooling_optim( self , p: _Array , rvY0: rv_base , rvXs: Sequence[rv_base] ) -> float:##{{{
        """Function to minimize to fit the alpha pooling"""
        if not self._fa:
            self.a = np.exp(p[0])
            self.w = np.exp(p[1:])
        else:
            self.w = np.exp(p)
        rvG    = self._pooling( rvXs )
        
        xmin = min( [rv.icdf(  1e-6) for rv in [rvY0,rvG]] )
        xmax = max( [rv.icdf(1-1e-6) for rv in [rvY0,rvG]] )
        x    = np.linspace( xmin , xmax , 1000 )
        
        return np.sum( np.abs(rvY0.cdf(x) - rvG.cdf(x))**2 )
    ##}}}
    
    def _fit_pooling( self , rvY0: rv_base , rvXs: Sequence[rv_base] ) -> None:##{{{
        """Function which initialize the fit of pooling, and fit"""
        
        ## Parameters
        nmod = len(rvXs)
        
        ## Find starting point
        if self._fa:
            x0    = np.log(np.ones(nmod))
        else:
            x0    = np.log(np.ones(nmod+1))
            x0[0] = np.log(0.5)
            s0    = self._fit_pooling_optim( x0 , rvY0 , rvXs )
            x0[0] = np.log(2)
            s1    = self._fit_pooling_optim( x0 , rvY0 , rvXs )
            x0[0] = np.log( 0.5 if s0 < s1 else 2 )
        
        ## Optimization
        success = False
        xn      = x0.copy()
        while not success:
            res     = sco.minimize( self._fit_pooling_optim , xn , args = (rvY0,rvXs) )
            if np.sum( np.abs(x0-xn) ) < 1e-3:
                break
            x0      = xn.copy()
            xn      = res.x
            success = res.success
        
        ## Set values
        if self._fa:
            self.w = np.exp(res.x)
        else:
            self.a = np.exp(res.x[0])
            self.w = np.exp(res.x[1:])
    ##}}}
    
    def fit( self , Y0: _Array , *args: Sequence[_Array] ) -> Self:##{{{
        """Fit the alpha-pooling class

        Parameters
        ----------
        Y0: numpy.ndarray
            Array of reference
        *args: Sequence[numpy.ndarray]
            List of biased models in calibration period, followed by the list
            of models in projection period
        
        Returns
        -------
        alp: SBCK.mm.AlphaPooling
            Alpha-pooling class fitted

        """
        
        ## Check number of arguments
        narg = len(args)
        if not narg % 2 == 0:
            raise ValueError( "Calibration and projection of each model must be given!")
        
        ## Split
        nmod = narg // 2
        X0s = args[:nmod]
        X1s = args[nmod:]
    
        ## Normalization
        NX1s,NX0s = self._normalization(Y0,X0s,X1s)
        
        ## Build random variables
        rvY0   =   self._typeY0.fit(Y0)
        rvX0s  = [ self._typeX0.fit(X0)  for  X0 in X0s  ]
        rvX1s  = [ self._typeX1.fit(X1)  for  X1 in X1s  ]
        rvNX0s = [ self._typeX0.fit(NX0) for NX0 in NX0s ]
        rvNX1s = [ self._typeX1.fit(NX1) for NX1 in NX1s ]
        
        ## Find alpha-pooling parameters
        self._fit_pooling( rvY0 , rvNX0s )
        
        ## Find 'future distribution of observations'
        self._rvY0  = self._pooling(rvNX0s)
        self._rvY1  = self._pooling(rvNX1s)
        Y1 = self._rvY1.rvs(10_000)
        
        ## And set multiple quantile mappings
        self._qm0 = []
        for rvX0,X0 in zip(rvX0s,X0s):
            self._qm0.append( QM().fit(Y0,X0) )
        self._qm1 = []
        for rvX1,X1 in zip(rvX1s,X1s):
            self._qm1.append( QM().fit(Y1,X1) )
        
        return self
    ##}}}
    
    def predict( self , *args: Sequence[_Array] ) -> Sequence[_Array]:##{{{
        """Predict the correction

        Return a tuple Z1s,Z0s if X0s it not None, else only Z1s.

        Parameters
        ----------
        *args: Sequence[numpy.ndarray]
            List of biased models in projection period, optionally followed by
            the list of models in calibration period
        
        Returns
        -------
        Z1s + Z0s: Sequence[numpy.ndarray]
            List of corrected models in projection period, optionnaly followed
            by the list of model in calibration period

        """
        
        ## Check and split calibration / projection
        nmod = len(self._qm0)
        if len(args) == nmod:
            X1s = args
            X0s = None
        elif len(args) == 2 * nmod:
            X1s = args[:nmod]
            X0s = args[nmod:]
        else:
            raise ValueError( "Projection of each model must be model, and optionally calibration of each model" )
        
        ## Correction
        Z1s = []
        for X1,qm1 in zip(X1s,self._qm1):
            Z1s.append(qm1.predict(X1))
        Z0s = None
        if X0s is not None:
            Z0s = []
            for X0,qm0 in zip(X0s,self._qm0):
                Z0s.append(qm0.predict(X0))
        
        ##
        if Z0s is not None:
            return Z1s + Z0s
        return Z1s
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def alpha(self) -> float:
        return self._a
    
    @property
    def a(self) -> float:
        return self._a
    
    @a.setter
    def a( self , a: float ) -> None:
        self._a = a
        
        x = np.linspace( 0 , 1 , 1000 )
        y = self._gpool(x)
        self._igpool = sci.interp1d( y , x , bounds_error = False , fill_value = (0,1) )
    
    @property
    def w(self) -> _Array:
        return self._w
    
    @w.setter
    def w( self , w: _Array ) -> None:
        w = np.array([w]).ravel()
        self._w = w / w.sum()
    
    ##}}}
    
##}}}

class AlphaPooling(MultiUBC):##{{{
    
    """AlphaPooling multi-model bias correction method (see [1]).
    
    Properties
    ----------
    alpha: _Array
        The alpha fitted for each marginal
    w: _Array
        The array of weights for each marginal
    
    Example
    -------
    >>> import SBCK
    >>>
    >>> ## Parameters
    >>> np.random.seed(42)
    >>> size    = 1_000
    >>> nmod    = 3
    >>> locs0   = np.linspace( -5 , 5 , nmod )
    >>> locs1   = np.linspace( 5 , 15 , nmod )
    >>> scales1 = np.linspace( 0.1 , 3 , nmod )
    >>> scales0 = np.linspace( 0.5 , 5 , nmod )
    >>> 
    >>> ## Data
    >>> Y0  =  np.random.normal( loc = 0 , scale = 1 , size = size )
    >>> X0s = [np.random.normal( loc = locs0[i] , scale = scales0[i] , size = size ) for i in range(nmod)]
    >>> X1s = [np.random.normal( loc = locs1[i] , scale = scales1[i] , size = size ) for i in range(nmod)]
    >>> 
    >>> ## mm correction, fix alpha
    >>> Z1s,Z0s = SBCK.mm.AlphaPooling( alpha = 3 ).fit( Y0 , X0s , X1s ).predict( X1s , X0s )
    >>> 
    >>> ## mm correction, fit alpha
    >>> Z1s,Z0s = SBCK.mm.AlphaPooling().fit( Y0 , X0s , X1s ).predict( X1s , X0s )
    
    References
    ----------
    [1] Vrac, M. and al: Distribution-based pooling for combination and
    multi-model bias correction of climate simulations, Earth Syst. Dynam., 15,
    735-762, doi:10.5194/esd-15-735-2024, 2024
    
    """
    
    def __init__( self ,  alpha: float | None = None , rvY0: rv_base = rv_empirical , rvX0: rv_base = rv_empirical , rvX1: rv_base = rv_empirical ) -> None:##{{{
        """
        Arguments
        ---------
        alpha: float | None
            The alpha parameter, if None, infered during the fit.
        rvY0: SBCK.stats.rv_base
            Law of references in calibration period
        rvX0: SBCK.stats.rv_base
            Law of model in calibration period
        rvX1: SBCK.stats.rv_base
            Law of model in projection period
        """
        
        ## And init upper class
        args   = tuple()
        kwargs = { "alpha" : alpha , "rvY0" : rvY0 , "rvX0" : rvX0 , "rvX1" : rvX1 }
        super().__init__( "AlphaPooling" , Univariate_AlphaPooling , args = args , kwargs = kwargs )
    ##}}}
    
    def fit( self , Y0: _Array , X0s : Sequence[_Array] , X1s: Sequence[_Array] ) -> Self:##{{{
        """Fit the alpha-pooling class

        Parameters
        ----------
        Y0: numpy.ndarray
            Array of reference
        X0s: Sequence[numpy.ndarray]
            List of biased models in calibration period
        X1s: Sequence[numpy.ndarray]
            List of biased models in projection period
        
        Returns
        -------
        alp: SBCK.mm.AlphaPooling
            Alpha-pooling class fitted

        """
        args = [Y0] + X0s + X1s
        return super().fit( *args )
    ##}}}
    
    def predict( self , X1s : Sequence[_Array] , X0s: Sequence[_Array] | None ) -> Sequence[_Array] | tuple[Sequence[_Array],Sequence[_Array]]:##{{{
        """Predict the correction

        Return a tuple Z1s,Z0s if X0s it not None, else only Z1s.

        Parameters
        ----------
        X1s: Sequence[numpy.ndarray]
            List of biased models in projection period
        X0s: Sequence[numpy.ndarray] | None
            List of biased models in calibration period, optional
        
        Returns
        -------
        Z1s: Sequence[numpy.ndarray]
            List of corrected models in projection period
        Z0s: Sequence[numpy.ndarray] | None
            List of correction models in calibration period, given only if
            X0s is not None

        """
        args = X1s
        if X0s is not None:
            args = args + X0s
        Z10s = super().predict(*args)
        if X0s is None:
            return Z10s
        else:
            Z1s = Z10s[:len(X1s)]
            Z0s = Z10s[len(X1s):]
        
        return Z1s,Z0s
    ##}}}

    ## Properties ##{{{
    
    @property
    def alpha(self) -> _Array:
        return np.array([ubcm.alpha for ubcm in self.ubcm]).ravel()
    
    @property
    def w(self) -> _Array:
        return np.array([ubcm.w for ubcm in self.ubcm])
    
    ##}}}
    
##}}}

