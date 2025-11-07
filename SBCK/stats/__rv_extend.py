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
import scipy.interpolate as sci


############
## Typing ##
############

from typing import Self
from typing import Sequence
from typing import Callable
from typing import Any
_rv_scipy        = sc._distn_infrastructure.rv_continuous
_rv_scipy_frozen = sc._distn_infrastructure.rv_continuous_frozen
_kernel_scipy    = sc._kde.gaussian_kde
_Array = np.ndarray


#####################
## Some generators ##
#####################

def rvs_spd_matrix( dim ):##{{{
    """A generator to draw random symetric positive definite matrix.

    Parameters
    ----------
    dim: int

    Returns
    -------
    S: numpy.ndarray
        A spd matrix
    """
    O = sc.ortho_group.rvs(dim)
    S = np.diag(np.random.exponential(size = dim))
    return O @ S @ O.T
##}}}


##############
## rv_class ##
##############

def io_type(func: Callable) -> Callable:##{{{
    """Decorator of the cdf, icdf and pdf method of rv_base used to cast input
    data in numpy.ndarray, and re-cast to original type in output.
    
    """
    def wrapper( self , x: Sequence[float] ) -> Sequence[float]:
        
        ## Transform to 1d array
        xt = np.atleast_1d( np.asarray(x) ).astype(float).ravel()
        
        ## Apply function
        yt = func( self , xt )
        
        ## And go back to x
        if np.isscalar(x):
            return float(yt[0])
        
        y = x.copy().astype(float) + np.nan
        y[:] = yt.reshape(y.shape)
        
        return y
    
    return wrapper
##}}}

class rv_base:##{{{
    """Base class of random variable, used to be derived.

    Properties
    ----------
    a: float
        Minimal value of the support of the rv
    b: float
        Maximal value of the support of the rv
    """
    
    _fcdf:  Callable | None
    _ficdf: Callable | None
    _fpdf:  Callable | None
    
    def __init__( self , cdf: Callable | None , icdf: Callable | None , pdf: Callable | None , *args , **kwargs ) -> None: ##{{{
        """
        Parameters
        ----------
        cdf: Callable | None
            The Cumulative Distribution Function
        icdf: Callable | None
            The inverse of the Cumulative Distribution Function
        pdf: Callable | None
            The Probability Density Function
        """
        self._fcdf  =  cdf
        self._ficdf = icdf
        self._fpdf  =  pdf
        pass
    ##}}}
    
    def rvs( self , size: int ) -> _Array :##{{{
        """Random value generator of the law

        Parameters
        ----------
        size: int
            Numbers of values to drawn

        Returns
        -------
        X: numpy.ndarray
            Data drawn
        """
        return self.icdf( np.random.uniform( size = size ) )
    ##}}}
    
    ## _cdf, _icdf and _pdf ##{{{
    
    def _cdf( self , x : _Array | Sequence[float] | float ) -> _Array:
        return self._fcdf(x)
    
    def _icdf( self , p : _Array | Sequence[float] | float ) -> _Array:
        return self._ficdf(p)
    
    def _pdf( self , x : _Array | Sequence[float] | float ) -> _Array:
        return self._fpdf(x)
    
    ##}}}
    
    @io_type
    def cdf( self , x: _Array ) -> _Array:##{{{
        """Cumulative Distribution Function

        Arguments
        ---------
        x: numpy.ndarray
            Quantiles

        Returns
        -------
        p: numpy.ndarray
            Probability to be lower than x
        """
        return self._cdf(x)
    ##}}}
    
    @io_type
    def icdf( self , p: _Array ) -> _Array:##{{{
        """Inverse of Cumulative Distribution Function

        Arguments
        ---------
        p: numpy.ndarray
            Probability to be lower than x

        Returns
        -------
        x: numpy.ndarray
            Quantiles
        """
        return self._icdf(p)
    ##}}}
    
    def sf( self , x: _Array ) -> _Array:##{{{
        """Survival Function

        Arguments
        ---------
        x: numpy.ndarray
            Quantiles

        Returns
        -------
        p: numpy.ndarray
            Probability to be greater than x
        """
        return 1 - self.cdf(x)
    ##}}}
    
    def isf( self , p: _Array ) -> _Array:##{{{
        """Inverse of Survival Function

        Arguments
        ---------
        p: numpy.ndarray
            Probability to be greater than x

        Returns
        -------
        x: numpy.ndarray
            Quantiles
        """
        return self.icdf(1-p)
    ##}}}
    
    def ppf( self , p: _Array ) -> _Array:##{{{
        """Inverse of Cumulative Distribution Function
        
        To be coherent with scipy

        Arguments
        ---------
        p: numpy.ndarray
            Probability to be lower than x

        Returns
        -------
        x: numpy.ndarray
            Quantiles
        """
        return self.icdf(p)
    ##}}}
    
    @io_type
    def pdf( self , x: _Array ) -> _Array:##{{{
        """Probability Density Function

        Arguments
        ---------
        x: numpy.ndarray
            Values

        Returns
        -------
        p: numpy.ndarray
            Density at x
        """
        return self._pdf(x)
    ##}}}
    
    @io_type
    def logpdf( self , x: _Array ) -> _Array:##{{{
        """Log of Probability Density Function

        Arguments
        ---------
        x: numpy.ndarray
            Values

        Returns
        -------
        p: numpy.ndarray
            Density at x
        """
        return np.log( self._pdf(x) )
    ##}}}

    def _init_icdf_from_cdf( self , xmin: float , xmax: float ) -> Callable:##{{{
        """Function used to find the inverse of the cdf

        Arguments
        ---------
        xmin: float
            Minimal value of the support of the cdf
        xmax: float
            Maximal value of the support of the cdf

        Returns
        -------
        icdf: Callable
            Inverse of CDF
        """
        
        x    = np.linspace( xmin , xmax , 1000 )
        p    = self._cdf(x)
        icdf = sci.interp1d( p , x , bounds_error = False , fill_value = (xmin,xmax) )
        
        return icdf
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def a(self) -> float:
        return self._icdf(0)
    
    @property
    def b(self) -> float:
        return self._icdf(1)
    ##}}}
    
##}}}

class rv_scipy(rv_base):##{{{
    
    _rvSC: _rv_scipy_frozen
    
    def __init__( self , rvSC: _rv_scipy_frozen ):##{{{
        """
        Parameters
        ----------
        rvSC: scipy.stats._distn_infrastructure.rv_continuous_frozen
            Frozen scipy rv
        """
        self._rvSC = rvSC
    ##}}}
    
    def _cdf( self , x: _Array ) -> _Array:##{{{
        return self._rvSC.cdf(x)
    ##}}}    
    
    def _icdf( self , p: _Array ) -> _Array:##{{{
        return self._rvSC.ppf(p)
    ##}}}
    
    def _pdf( self , x: _Array ) -> _Array:##{{{
        return self._rvSC.pdf(x)
    ##}}}
    
    @staticmethod
    def fit( X: _Array , type_: _rv_scipy ) -> Self:##{{{
        """Static fit method
    
        Arguments
        ---------
        X: numpy.ndarray
            Data to fit the law
        type_: scipy.stats._distn_infrastructure.rv_continuous
            Scipy type of the law

        Returns
        -------
        rvSC: SBCK.stats.rv_scipy
            The law fitted
        """
        return rv_scipy( type_( *type_.fit(X) ) )
    ##}}}

##}}}

class rv_empirical(rv_base):##{{{
    """Empirical histogram class. The differences with scipy.stats.rv_histogram
    are 1. the fit method and 2. the way to infer the cdf and the icdf. Here:
    
    >>> X ## Input
    >>> rvX = rv_empirical.fit(X)
    """
    
    def __init__( self , *args: Callable , **kwargs ) -> None:##{{{
        """
        Arguments
        ---------
        args: Callable
            All arguments are passed to SBCK.stats.rv_base
        kwargs: Callable
            All keywords arguments are passed to SBCK.stats.rv_base
        """
        super().__init__( *args , **kwargs )
    ##}}}
    
    @staticmethod
    def fit( X: _Array , *args: Any , bins: int | str = "auto" , bins_min: int = 21 , bins_max: int = 101 , **kwargs: Any ) -> Self:##{{{
        """Static fit method
        
        Arguments
        ---------
        X: numpy.ndarray
            Init the rv_empirical with X
        bins: int | str = "auto" int(0.1*X.size)
            Numbers of bin used to infer the pdf. If 'auto', 0.1 * X.size
            is used, and values must be between bins_min and bins_max
        bin_min: int = 21
            Minimal numbers of bin used to infer the pdf
        bin_max: int = 101
            Maximal numbers of bin used to infer the pdf
        
        
        Returns
        -------
        rvX: SBCK.stats.rv_empirical
            Random variable initialized
        """
        
        Xs = np.sort(X.squeeze())
        Xr = sc.rankdata(Xs,method="max")
        p  = np.unique(Xr) / X.size
        q  = Xs[np.unique(Xr)-1]
        
        if p[0] > 0:
            q  = np.hstack( (q[0] - np.sqrt(np.finfo(float).resolution),q) )
            p  = np.hstack( (0,p) )
        
        icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (q[0],q[-1]) )
        cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (0,1) )
        
        dq   = 0.05 * (q.max() - q.min())
        if bins == "auto":
            bins = int(0.1 * X.size)
        bins = np.linspace( q.min() - dq , q.max() + dq , min( max( bins , bins_min ) , bins_max ) )
        h,c  = np.histogram( X , bins , density = True )
        c    = (c[1:] + c[:-1]) / 2
        pdf  = sci.interp1d( c , h , bounds_error = False , fill_value = (0,0) ) 
        
        return rv_empirical( cdf , icdf , pdf )
    ##}}}
    
##}}}

class rv_empirical_ratio(rv_empirical):##{{{
    """Extension of SBCK.stats.rv_empirical taking into account of a "ratio" part,
    i.e., instead of fitting:
    P( X < x )
    We fit separatly the frequency of 0 and:
    P( X < x | X > 0 )
    """
    
    _p0: float
    
    def __init__( self , p0: float , *args , **kwargs ) -> None:##{{{
        """
        Arguments
        ---------
        p0: float | None
            The probability at 0
        args: Callable
            All arguments are passed to SBCK.stats.rv_base
        kwargs: Callable
            All keywords arguments are passed to SBCK.stats.rv_base
        """
        
        self._p0  = p0
        super().__init__( *args , **kwargs )
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def p0(self) -> float:
        return self._p0
    
    ##}}}
    
    @staticmethod
    def fit( X: _Array , *args: Any , **kwargs: Any ) -> Self:##{{{
        """Fit method
        
        Arguments
        ---------
        X: numpy.ndarray
            Init the rv_empirical_ratio with X
        
        Returns
        -------
        rvX: SBCK.stats.rv_empirical_ratio
            Random variable initialized
        """
        Xp = X[X>0]
        p0 = 1 - np.sum( X>0 ) / X.size
        rvXp = rv_empirical.fit(Xp)
        return rv_empirical_ratio( p0 , rvXp._fcdf , rvXp._ficdf , rvXp._fpdf )
    ##}}}
    
    def _cdf( self , x: _Array ) -> _Array:##{{{
        p = np.zeros_like(x) + np.nan
        idxp = x > 0
        idxn = x < 0
        idx0 = ~idxp & ~idxn
        p[idxp] = (1-self.p0) * super()._cdf( x[idxp] ) + self.p0
        p[idx0] = self.p0
        p[idxn] = 0
        return p
    ##}}}
    
    def _icdf( self , p: _Array ) -> _Array:##{{{
        
        x    = np.zeros_like(p) + np.nan
        idxp = p > self.p0
        idx0 = ~idxp
        x[idxp] = super()._icdf( (p[idxp] - self.p0) / (1-self.p0) )
        x[idx0] = 0
        return x
    ##}}}
    
    def _pdf( self , x: _Array ) -> _Array:##{{{
        
        pdf = np.zeros_like(x) + np.nan
        xp  = x[x>0].min() / 2
        xn  = x[x<0].max() / 2
        
        idxp = x > xp
        idxn = x < xn
        idx0 = ~idxp & ~idxn
        
        pdf[idxn] = 0
        pdf[idxp] = super()._pdf(x[idxp]) * (1 - self.p0)
        pdf[idx0] = sc.uniform.pdf( x[idx0] , loc = xp , scale = xn ) * self.p0
        return pdf
    ##}}}
    
##}}}

class rv_empirical_gpd(rv_empirical):##{{{
    """Empirical histogram class where tails are given by the fit of
    Generalized Pareto Distribution.
    """
    
    _locl: float
    _locr: float
    _gpdl: rv_scipy
    _gpdr: rv_scipy
    _pl: float
    _pr: float
    
    def __init__( self , locl: float, locr: float, gpdl: rv_scipy, gpdr: rv_scipy, pl: float, pr: float, *args: Callable , **kwargs) -> None: ##{{{
        """
        Arguments
        ---------
        locl: float
            Location parameter of the left GPD
        locr: float
            Location parameter of the right GPD
        gpdl: SBCK.stats.rv_scipy
            Law of the left tail
        gpdr: SBCK.stats.rv_scipy
            Law of the right tail
        pl: float
            Probability of the left tail
        pr: float
            Probability of the right tail
        args: Callable
            All arguments are passed to SBCK.stats.rv_base
        kwargs: Callable
            All keywords arguments are passed to SBCK.stats.rv_base
        """
        
        super().__init__( *args , **kwargs )
        self._locl = locl
        self._locr = locr
        self._gpdl = gpdl
        self._gpdr = gpdr
        self._pl   = pl
        self._pr   = pr
    ##}}}
    
    @staticmethod
    def fit( X: _Array , *args: Any , pl: float = 0.05 , pr: float = 1 - 0.05 , **kwargs: Any ) -> Self:##{{{
        """
        Arguments
        ---------
        X: np.ndarray | None
            Init the rv_empirical_gpd with X
        pl: float = 0.05
            Probability of the left tail
        pr: float = 1 - 0.05
            Probability of the right tail
        """
        
        ## Location parameter
        locl,locr = np.quantile( X , [pl,pr] )
        
        ## Cut X in the three part
        idxl = X < locl
        idxr = X > locr
        idxm = ~idxl & ~idxr
        Xl   = X[idxl] - locl
        Xr   = X[idxr] - locr
        Xm   = X[idxm]
        
        ## Empirical part
        rvXm = rv_empirical.fit(Xm)
        
        def gpdfit(X):
            lmom  = sc.lmoment( X , order = [1,2] )
            itau  = lmom[0] / lmom[1]
            scale = lmom[0] * ( itau - 1 )
            scale = scale if scale > 0 else 1e-8
            shape = 2 - itau
            return scale,shape
        
        ## GPD left fit
        scl,shl = gpdfit(-Xl)
        gpdl    = rv_scipy( sc.genpareto( loc = 0 , scale = scl , c = shl ) )
        
        ## GPD right fit
        scr,shr = gpdfit(Xr)
        gpdr    = rv_scipy( sc.genpareto( loc = 0 , scale = scr , c = shr ) )
        
        return rv_empirical_gpd( locl , locr , gpdl , gpdr , pl , pr , rvXm._fcdf , rvXm._ficdf , rvXm._fpdf )
    ##}}}
    
    def _pdf( self , x: _Array ) -> _Array:##{{{
        
        pdf = np.zeros_like(x)
        
        ## Split index into left, middle and right part
        idxl = x < self.locl
        idxr = x > self.locr
        idxm = ~idxl & ~idxr
        
        ## Empirical
        pdf[idxm] = rv_empirical._pdf( self , x[idxm] ) * (self.pr - self.pl)
        
        ## Left part
        pdf[idxl] = self.pl * self._gpdl.pdf( -(x[idxl] - self.locl) )
        
        ## Right part
        pdf[idxr] = self._gpdr.pdf(x[idxr] - self.locr) * (1-self.pr)
        
        return pdf
    ##}}}
    
    def _cdf( self , x: _Array ) -> _Array:##{{{
        
        p = np.zeros_like(x) + np.nan
        
        ## Split index into left, middle and right part
        idxl = x < self.locl
        idxr = x > self.locr
        idxm = ~idxl & ~idxr
        
        ## Empirical
        p[idxm] = rv_empirical._cdf( self , x[idxm] ) * (self.pr - self.pl) + self.pl
        
        ## Left part
        p[idxl] = self.pl * self._gpdl.sf( -(x[idxl] - self.locl) )
        
        ## Right part
        p[idxr] = self.pr + (1-self.pr) * self._gpdr.cdf(x[idxr] - self.locr)
        
        return p
    ##}}}
    
    def _icdf( self , p: _Array ) -> _Array:##{{{
        
        x = np.zeros_like(p) + np.nan
        
        ## Split index into left, middle and right part
        idxl = p < self.pl
        idxr = p > self.pr
        idxm = ~idxl & ~idxr
        
        ## Empirical part
        x[idxm] = rv_empirical._icdf( self , ( p[idxm] - self.pl ) / (self.pr - self.pl) )
        
        ## Left part
        x[idxl] = self.locl - self._gpdl.isf( p[idxl] / self.pl )
        
        ## Right part
        x[idxr] = self.locr + self._gpdr.ppf( (p[idxr] - self.pr) / (1-self.pr ))
        
        return x
    ##}}}
    
    ## Attributes ##{{{
    
    @property
    def pl(self) -> float:
        return self._pl
    
    @property
    def pr(self) -> float:
        return self._pr
    
    @property
    def locl(self) -> float:
        return self._locl
    
    @property
    def locr(self) -> float:
        return self._locr
    
    @property
    def scalel(self) -> float:
        return float(self._gpdl.kwds["scale"])
    
    @property
    def scaler(self) -> float:
        return float(self._gpdr.kwds["scale"])
    
    @property
    def shapel(self) -> float:
        return float(self._gpdl.kwds["c"])
    
    @property
    def shaper(self) -> float:
        return float(self._gpdr.kwds["c"])
    
    ##}}}
    
##}}}

class rv_density(rv_base):##{{{
    """Empirical density class. Use a gaussian kernel. 'bw_method' argument
    can be given as kwargs to scipy.stats.gaussian_kde
    
    """
    
    _kernel: _kernel_scipy

    def __init__( self , kernel: _kernel_scipy , *args , **kwargs ) -> None:##{{{
        """
        Arguments
        ---------
        kernel: scipy.stats._kde.gaussian_kde
            Kernel fitted by scipy
        args: Callable
            All arguments are passed to SBCK.stats.rv_base
        kwargs: Callable
            All keywords arguments are passed to SBCK.stats.rv_base
        """
        self._kernel = kernel
        icdf = self._init_icdf_from_cdf( self._kernel.dataset.min() , self._kernel.dataset.max() )
        super().__init__( None , icdf , None , **kwargs )
        
    ##}}}
    
    def fit( X: _Array , *args: Any , bw_method: str | float | Callable | None = None , **kwargs: Any ) -> Self:##{{{
        """
        Arguments
        ---------
        X: numpy.ndarray
            Init the rv_density with X
        bw_method:
            bandwidth method, see scipy.stats.gaussian_kde
        
        Returns
        -------
        rv: SBCK.stats.rv_density
        """
        return rv_density( sc.gaussian_kde( X , bw_method = bw_method ) )
    ##}}}
    
    def _cdf( self , x: _Array ) -> _Array:##{{{
        cdf = np.apply_along_axis( lambda z: self._kernel.integrate_box_1d( -np.inf , z ) , 1 , x.reshape(-1,1) )
        cdf[cdf < 0] = 0
        cdf[cdf > 1] = 1
        return cdf.squeeze()
    ##}}}
    
    def _pdf( self , x: _Array ) -> _Array:##{{{
        return self._kernel.pdf(x)
    ##}}}
    
##}}}

class rv_mixture(rv_base):##{{{
    """Mixture of distributions. The fit method raise a NotImplementedError
    """
    
    _dist: list[rv_base] = []
    _weights: np.ndarray
    
    def __init__( self , *args: rv_base | _rv_scipy_frozen  , weights: list | np.ndarray | None = None , **kwargs ):##{{{
        """
        Arguments
        ---------
        args:
            A list of rv_base based or frozen scipy distribution
        weights:
            The weight, default is uniform
        """
        
        
        ## Init laws
        self._dist = []
        for d in args:
            if isinstance( d, rv_base ):
                self._dist.append(d)
            elif isinstance( d , _rv_scipy_frozen ):
                self._dist.append( rv_scipy(d) )
            else:
                raise ValueError(f"Unknow type of {d}")
        ndist = len(self._dist)
        self._weights = None
        
        if weights is None:
            self._weights = np.ones(ndist)
        else:
            self._weights = np.array([weights]).ravel()
        self._weights /= self._weights.sum()
        
        ## Build the icdf
        xmin =  np.inf
        xmax = -np.inf
        for d in self._dist:
            xmin = min( xmin , d.icdf(  1e-3) )
            xmax = max( xmax , d.icdf(1-1e-3) )
        icdf = self._init_icdf_from_cdf(xmin,xmax)
        
        super().__init__( None , icdf , None )
    ##}}}
    
    def _pdf( self , x: _Array ) -> _Array:##{{{
        pdf = np.zeros_like(x)
        for d,w in zip(self._dist,self._weights):
            pdf += w * d.pdf(x)
        return pdf
    ##}}}
    
    def _cdf( self , x: _Array ) -> _Array:##{{{
        cdf = np.zeros_like(x)
        for d,w in zip(self._dist,self._weights):
            cdf += w * d.cdf(x)
        return cdf
    ##}}}
    
    @staticmethod
    def fit( X: np.ndarray , *args: rv_base | _rv_scipy , **kwargs: Any ) -> Self:##{{{
        """
        Use a MLE method, but not work currentlu
        
        Arguments
        ---------
        X: numpy.ndarray
            Init the rv_density with X
        args:
            A list of scipy.stats.* distribution, not initialized
        
        Returns
        -------
        """
        raise NotImplementedError
#
#        ## Step 1, find the numbers of parameters
#        nhpars = []
#        x0     = []
#        for law in args:
#            ehpar = list(law.fit(X))
#            x0 = x0 + ehpar
#            nhpars.append( len(ehpar) )
#        thpar = sum(nhpars)
#        
#        ## Define the likelihood
#        def mle( x , X , nhpars , *args ):
#            
#            ## Extract hyper-parameters and weights
#            thpar = sum(nhpars)
#            hpars = x[:thpar]
#            ws    = np.exp(x[thpar:])
#            ws    = ws / ws.sum()
#            
#            ## Split hpar in hpar for each law
#            i0,i1  = 0,0
#            lhpars = []
#            for n in nhpars:
#                i1 = i0 + n
#                lhpars.append(hpars[i0:i1])
#                i0 = i1
#            
#            ## And compute likelihood
#            ll = 0
#            for law,hpar,w in zip(args,lhpars,ws):
#                ll += w * law.logpdf( X , *hpar.tolist() ).sum()
#            
#            return -ll
#        
#        ## Define the init point
#        x0 = np.array( [x0 + np.zeros(len(args)).tolist()] ).ravel()
#        
#        ## Optimization
#        res = sco.minimize( mle , x0 , args = (X,nhpars) + args , method = "Nelder-Mead" )
#        x   = res.x
#        
#        ## Split output parameters
#        hpars = x[:thpar]
#        ws    = np.exp(x[thpar:])
#        ws    = ws / ws.sum()
#        i0,i1  = 0,0
#        lhpars = []
#        for n in nhpars:
#            i1 = i0 + n
#            lhpars.append(hpars[i0:i1])
#            i0 = i1
#        
#        ##
#        oargs = []
#        for law,hpar in zip(args,lhpars):
#            oargs.append( law( *hpar.tolist() ) )
#        oargs.append(ws)
#        
#        return tuple(oargs)
    ##}}}
    
##}}}


###########################
## Multivariate rv_class ##
###########################

class mrv_base:##{{{
    """Class used to transform univariate rv in multivariate rv. Each margins
    is fitted separately.
    """
    
    _dist: list[rv_base]
    
    def __init__( self , *args: rv_base | _rv_scipy_frozen ) -> None:##{{{
        """
        Parameters
        ----------
        args: rv_base | scipy.stats._distn_infrastructure.rv_continuous_frozen
            Law of each marginal
        """
        self._dist = []
        for d in args:
            if isinstance( d, rv_base ):
                self._dist.append(d)
            elif isinstance( d , _rv_scipy_frozen ):
                self._dist.append( rv_scipy(d) )
            else:
                raise ValueError(f"Unknow type of {d}")
    ##}}}
    
    @staticmethod
    def fit( X: _Array , *args: rv_base | _rv_scipy ) -> Self:##{{{
        """
        Parameters
        ----------
        X: numpy.ndarray
            Data to fit
        args: rv_base | scipy.stats._distn_infrastructure.rv_continuous_frozen
            Law of each marginal

        Returns
        -------
        rv: mrv_base
            Law fitted
        """
        
        ## Dimensions of X
        if X.ndim == 1:
            X = X.reshape(-1,1)
        ndim = X.shape[1]
        
        ## distributions
        dist = args
        if len(args) == 0:
            dist = [ rv_empirical for _ in range(ndim) ]
        if len(args) == 1 and ndim > 1:
            dist = [ args[0] for _ in range(ndim) ]
        if not len(dist) == ndim:
            raise ValueError("Incoherent size between X and law parameters")
        ## Now fit
        fdist = []
        for i,d in enumerate(dist):
            if issubclass( d , rv_base ):
                fdist.append( d.fit(X[:,i]) )
            elif isinstance( d , _rv_scipy ):
                fdist.append( rv_scipy.fit( X[:,i] , d ) )
            else:
                raise ValueError(f"Unknow distribution {d}")
        
        return mrv_base( *fdist )
    ##}}}
    
    ## Statistical methods ##{{{
    
    def rvs( self , size: int = 1 ) -> _Array:
        """Random value generator of the law

        Parameters
        ----------
        size: int
            Numbers of values to drawn

        Returns
        -------
        X: numpy.ndarray
            Data drawn
        """
        return np.array( [ d.rvs(size=size) for d in self._dist ] ).T.copy()
    
    def cdf( self , x: _Array ) -> _Array:
        """Cumulative Distribution Function

        Arguments
        ---------
        x: numpy.ndarray
            Quantiles

        Returns
        -------
        p: numpy.ndarray
            Probability to be lower than x
        """
        x = x.reshape(-1,self.ndim)
        return np.array( [ d.cdf(x[:,i]) for i,d in enumerate(self._dist) ] ).T.copy()
    
    def sf( self , x: _Array ) -> _Array:
        """Survival Function

        Arguments
        ---------
        x: numpy.ndarray
            Quantiles

        Returns
        -------
        p: numpy.ndarray
            Probability to be greater than x
        """
        return 1 - self.cdf(x)
    
    def icdf( self , p: _Array ) -> _Array:
        """Inverse of Cumulative Distribution Function

        Arguments
        ---------
        p: numpy.ndarray
            Probability to be lower than x

        Returns
        -------
        x: numpy.ndarray
            Quantiles
        """
        p = p.reshape(-1,self.ndim)
        return np.array( [ d.icdf(p[:,i]) for i,d in enumerate(self._dist) ] ).T.copy()
    
    def isf( self , p: _Array ) -> _Array:
        """Inverse of Survival Function

        Arguments
        ---------
        p: numpy.ndarray
            Probability to be greater than x

        Returns
        -------
        x: numpy.ndarray
            Quantiles
        """
        return self.icdf(1-p)
    
    def ppf( self , p: _Array ) -> _Array:
        """Inverse of Cumulative Distribution Function
        
        To be coherent with scipy

        Arguments
        ---------
        p: numpy.ndarray
            Probability to be lower than x

        Returns
        -------
        x: numpy.ndarray
            Quantiles
        """
        return self.icdf(p)
    
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def ndim(self) -> int:
        return len(self._dist)
    ##}}}
    
##}}}


