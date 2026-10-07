
## Copyright(c) 2022 / 2026 Yoann Robin
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
from .__PrePostProcessing import PrePostProcessingPerCols
from ..misc.__sys import deprecated


############
## Typing ##
############

from typing import Any
from typing import Callable

_Array = np.ndarray


###########
## Class ##
###########

class LinkFunction(PrePostProcessingPerCols):##{{{
    """This class is used to define pre/post processing class with a link
    function and its inverse. See also the PrePostProcessing documentation
    
    >>> ## Start with data
    >>> Y0,X0,X1 = SBCK.datasets.like_tas_pr(2000)
    >>> 
    >>> ## Define the link function
    >>> transform  = lambda x : x**3
    >>> itransform = lamnda x : x**(1/3)
    >>> 
    >>> ## And the PPP method
    >>> ppp = SBCK.ppp.LinkFunction( bc_method = SBCK.CDFt ,
    >>>                                transform_ = transform ,
    >>>                               itransform_ = itransform )
    >>> 
    >>> ## And now the correction
    >>> ## Bias correction
    >>> ppp.fit(Y0,X0,X1)
    >>> Z = ppp.predict(X1,X0)
    
    """
    
    _f_transform: Callable
    _f_itransform: Callable
    
    def __init__( self , *args: Any , name: str = "LinkFunction" , transform_: Callable | None = None , itransform_: Callable | None = None, **kwargs: Any ) -> None:##{{{
        """
        Arguments
        ---------
        name: str
            Name of the link function
        transform_: Callable
            Function to transform the data
        itransform_: Callable
            Function to inverse the transform of the data
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        super().__init__( *args , **kwargs )
        self._name         = name
        self._f_transform  = transform_
        self._f_itransform = itransform_
    
    ##}}}
    
    ## Transform and itransform functions ##{{{
    
    def _transform( self , X: _Array ) -> _Array:
        return self._f_transform(X)
    
    def _itransform( self , Xt: _Array ) -> _Array:
        return self._f_itransform(Xt)
    
    def transform( self , X: _Array ) -> _Array:
        """
        Apply the transform
        """
        Xt = X.copy()
        Xt[:,self.cols] = self._transform(X[:,self.cols])
        return Xt
    
    def itransform( self , Xt: _Array ) -> _Array:
        """
        Apply the inverse transform
        """
        X = Xt.copy()
        X[:,self.cols] = self._itransform(Xt[:,self.cols])
        return X
    ##}}}
    
##}}}

class LFAdd(LinkFunction):##{{{
    """Addition link transform.
    """
    def __init__( self, m: float, *args: Any, **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        m : float
            The value to add
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        
        transform  = lambda x: x + m
        itransform = lambda x: x - m
        LinkFunction.__init__( self , *args, name = "LFAdd", transform_ = transform, itransform_ = itransform, **kwargs )
##}}}

class LFMult(LinkFunction):##{{{
    """Multiplication link transform.
    """
    
    def __init__( self, s: float, *args: Any, **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        s : float
            The value to multiply
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        
        transform  = lambda x: x * s
        itransform = lambda x: x / s
        LinkFunction.__init__( self, *args, name = "LFMult", transform_ = transform, itransform_ = itransform, **kwargs )
##}}}

class LFMax(LinkFunction):##{{{
    """Max link.
    """
    
    def __init__( self, M: float, *args: Any, **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        M : float
            The max
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        
        transform  = lambda x: np.where( (x < M) | ~np.isfinite(x) , x , M )
        itransform = lambda x: np.where( (x < M) | ~np.isfinite(x) , x , M )
        LinkFunction.__init__( self, *args, name = "LFMax", transform_ = transform, itransform_ = itransform, **kwargs )
##}}}

class LFMin(LinkFunction):##{{{
    """Min link.
    """
    
    def __init__( self, M: float, *args: Any, **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        M : float
            The min
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        
        transform  = lambda x: np.where( (x > M) | ~np.isfinite(x) , x , M )
        itransform = lambda x: np.where( (x > M) | ~np.isfinite(x) , x , M )
        LinkFunction.__init__( self, *args, name = "LFMin", transform_ = transform, itransform_ = itransform, **kwargs )
##}}}

class LFSquare(LinkFunction):##{{{
    """Square link transform, i.e.:
    - transform is given by lambda x: x**2
    - inverse transform is given by lambda x: sign(x) * sqrt(abs(x))
    """
    
    def __init__( self, *args: Any, **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        transform  = lambda x : x**2
        itransform = lambda x : np.where( x > 0 , np.sqrt(np.abs(x)) , - np.sqrt(np.abs(x)))
        LinkFunction.__init__( self, *args, name = "LFSquare", transform_ = transform, itransform_ = itransform, **kwargs )
##}}}

class LFLog(LinkFunction):##{{{
    """Log link transform, i.e.:
    - transform is given by log(x)
    - inverse transform is given by exp(x)
    
    """
    
    def __init__( self, *args: Any, **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        transform  = lambda x: np.log(x)
        itransform = lambda x: np.exp(x)
        LinkFunction.__init__( self, *args, name = "LFLoglin", transform_ = transform, itransform_ = itransform, **kwargs )
    
##}}}

class LFLoglin(LinkFunction):##{{{
    """Log linear link transform, i.e.:
    - transform is given by s*log(x/s) + s if 0 < x < s, else x
    - inverse transform is given by s*exp( (x-s) / s ) if x < s, else x
    
    """
    
    def __init__( self, *args: Any, s: float = 1e-5, **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        s: float
            Value where the exponential is transformed to identity
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        if not s > 0:
            raise Exception( f"Parameter s = {s} must be non negative!" )
        self.s = s
        transform  = lambda x: np.where( (0 < x) & (x < s) , s * np.log( np.where( x > 0 , x , np.nan ) / s ) + s , np.where( x < 0 , np.nan , x ) )
        itransform = lambda x: np.where( x < s , s * np.exp( (x-s) / s ) , x )
        LinkFunction.__init__( self, *args, name = "LFLoglin", transform_ = transform, itransform_ = itransform, **kwargs )
    
##}}}

class LFArctan(LinkFunction):##{{{
    """Arctan link transform, to bound the correction between two values.
    """
    
    def __init__( self, ymin: float, ymax: float, *args: Any, **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        ymin: float
            Minimal value
        ymax: float
            Maximal value
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        
        f = (ymax - ymin) / np.pi
        transform  = lambda x: np.where( (x > ymin) & (x < ymax) , f * np.tan( (x - ymin) / f - np.pi / 2 ) , np.nan )
        itransform = lambda x: (np.pi / 2 + np.arctan(x/f) ) * f + ymin
        LinkFunction.__init__( self, *args, name = "LFArctan", transform_ = transform, itransform_ = itransform, **kwargs )
##}}}

class LFLogistic(LinkFunction):##{{{
    """Logistic link transform, to bound the correction between two values.
    Starting from a dataset bounded between ymin and ymax, the transform maps
    the interval [ymin,ymax] to R with:
    
    transform : x |-> - np.log( (ymax - ymin) / (x - ymin) - 1 ) / s
    
    and the inverse transform is the logistic function:
    
    itransform : y |-> (ymax - ymin) / ( 1 + np.exp(-s*y) ) + ymin
    """
    
    def __init__( self, ymin: float, ymax: float, *args: Any, s: float = 1, tol: float = 1e-8, **kwargs: Any ) -> None:
        """
        Arguments
        ---------
        ymin: float
            Minimal value
        ymax: float
            Maximal value
        s: float
            The slope around 0 of the transform, default to 1
        tol: float
            Numerical tolerance
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessingPerCols
        *kwargs:
            All others arguments are passed to
            SBCK.ppp.PrePostProcessingPerCols, including:
            cols: Sequence[int] | int | np.ndarray[int] | slice = slice(None)
                The columns to apply
        """
        
        self.ymin = ymin
        self.ymax = ymax
        self.s    = s
        self._tol = tol
        
        LinkFunction.__init__( self, *args, name = "LFLogistic", **kwargs )
    
    def _transform( self , x: _Array ) -> _Array:
        xt  = x.copy()
        xt  = np.where( xt < self.ymax - self._tol , xt , self.ymax - self._tol )
        xt  = np.where( xt > self.ymin + self._tol , xt , self.ymin + self._tol )
        y   = - np.log( (self.ymax - self.ymin) / (xt - self.ymin) - 1 ) / self.s
        ivl = (x < self.ymin) | (x > self.ymax)
        y[ivl] = np.nan
        return y
    
    def _itransform( self , y: _Array ) -> _Array:
        x = y.copy()
        x = (self.ymax - self.ymin) / ( 1 + np.exp(-self.s*x) ) + self.ymin
        x = np.where( x < self.ymax - self._tol , x , self.ymax )
        x = np.where( x > self.ymin + self._tol , x , self.ymin )
        return x
    
##}}}


######################
## Deprecated names ##
######################

@deprecated( "PPPLinkFunction is renamed LinkFunction since the version 2.0.0" )
class PPPLinkFunction(LinkFunction):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPLinkFunction"
    ##}}}
    
##}}}

@deprecated( "PPPAddLink is renamed LFAdd since the version 2.0.0" )
class PPPAddLink(LFAdd):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPAddLink"
    ##}}}
    
##}}}

@deprecated( "PPPMultLink is renamed LFMult since the version 2.0.0" )
class PPPMultLink(LFMult):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPMultLink"
    ##}}}
    
##}}}

@deprecated( "PPPMaxLink is renamed LFMax since the version 2.0.0" )
class PPPMaxLink(LFMax):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPMaxLink"
    ##}}}
    
##}}}

@deprecated( "PPPMinLink is renamed LFMin since the version 2.0.0" )
class PPPMinLink(LFMin):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPMinLink"
    ##}}}
    
##}}}

@deprecated( "PPPSquareLink is renamed LFSquare since the version 2.0.0" )
class PPPSquareLink(LFSquare):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPSquareLink"
    ##}}}
    
##}}}

@deprecated( "PPPLogLinLink is renamed LFLoglin since the version 2.0.0" )
class PPPLogLinLink(LFLoglin):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPLogLinLink"
    ##}}}
    
##}}}

@deprecated( "PPPArctanLink is renamed LFArctan since the version 2.0.0" )
class PPPArctanLink(LFArctan):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPArctanLink"
    ##}}}
    
##}}}

@deprecated( "PPPLogisticLink is renamed LFLogistic since the version 2.0.0" )
class PPPLogisticLink(LFLogistic):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPLogisticLink"
    ##}}}
    
##}}}

