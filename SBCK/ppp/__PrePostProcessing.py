
## Copyright(c) 2022 / 2025 Yoann Robin
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

from ..__AbstractBC import AbstractBC
from ..__miscBC import IdBC
from .__checkf import allfinite


############
## Typing ##
############

from typing import Sequence
from typing import Callable
from typing import Any
from typing import Self

_Array = np.ndarray
_NArray = _Array | None


###########
## Class ##
###########

class PrePostProcessing(AbstractBC):##{{{
    
    """
    SBCK.ppp.PrePostProcessing
    ==========================
    
    This base class can be considered as the identity pre-post processing, and
    is used to be herited by others pre/post processing class. The key ideas are:
    - A PrePostProcessing based class contains a bias correction method, initalized
      by the `bc_method` argument, always available for all herited class
    - The `pipe` keyword is a list of pre/post processing class, applied one after
      the other.
    
    Try with an example, start with a dataset similar to tas/pr:
    >>> Y0,X0,X1 = SBCK.datasets.like_tas_pr(2000)
    
    The first column is Gaussian, but the second is an exponential law with a Dirac
    mass at 0, represented the 0 of precipitations. For a quantile mapping
    correction in the calibration period, we just apply:
    >>> qm = SBCK.QM()
    >>> qm.fit(Y0,X0)
    >>> Z0 = qm.predict(X0)
    
    Now, if we want to pre-post process with the SSR method (0 are replaced by
    random values between 0 (excluded) and the minimal non zero value), we write:
    >>> ppp = SBCK.ppp.SSR( bc_method = SBCK.QM , cols = [2] )
    >>> ppp.fit(Y0,X0)
    >>> Z0 = ppp.predict(X0)
    
    The SSR approach is applied only on the second column (the precipitation), and
    the syntax is the same than for a simple bias correction method.
    
    Imagine now that we want to apply the SSR, and to ensure the positivity of CDFt
    for precipitation, we also want to use the LogLinLink pre-post processing
    method. This can be done with the following syntax:
    >>> ppp = SBCK.ppp.LFLoglin( bc_method = SBCK.CDFt , cols = [2] ,
    >>>                          pipe = [SBCK.ppp.SSR] ,
    >>>                          pipe_kwargs = [{"cols" : 2}] )
    >>> ppp.fit(Y0,X0,X1)
    >>> Z = ppp.predict(X1,X0)
    
    With this syntax, the pre processing operation is
    LFLoglin.transform(SSR.transform(data)) and post processing operation
    SSR.itransform(LFLoglin.itransform(bc_data)). So the formula can read
    from right to left (as the mathematical composition). Note it is equivalent
    to define:
    >>> ppp = SBCK.ppp.PrePostProcessing( bc_method = SBCK.CDFt,
    >>>                     pipe = [SBCK.ppp.LFLoglin,SBCK.ppp.SSR],
    >>>                     pipe_kwargs = [ {"cols":2} , {"cols":2} ] )
    
    """
    
    _pipe: Sequence[AbstractBC]
    _bc_method: AbstractBC
    _kind: str | None
    _checkf: Callable
    _check: bool | None
    
    def __init__( self , bc_method: AbstractBC = None , bc_method_kwargs: dict[Any] = {} , pipe: Sequence[AbstractBC] = [] , pipe_kwargs: Sequence[dict[Any]] = [] , checkf: Callable = allfinite ) -> None:##{{{
        """
        Arguments
        ---------
        bc_method: [SBCK.<Bias Correction class]
            A bias correction method, optional if this class is given in 'pipe'
        bc_method_kwargs: dict
            Keyword arguments given to the constructor of bc_method
        pipe: list of PrePostProcessing class
            List of preprocessing class to apply to data before / after bias
            correction.
        pipe_kwargs: list of dict
            List of keyword arguments to pass to each PreProcessing class. This
            argument must either be an empty list, or a list of the same length
            as 'pipe'.
        checkf: Callable
            Boolean function controlling if the fit can occurs. Intercept
            'bc_method' and 'pipe' before their applications. If check return
            False on the dataset fitted, the fit doesn't occurs, and the predict
            return the input.
        """
        
        super().__init__( "PrePostProcessing" , "None" )
        
        if not isinstance( pipe , (list,tuple) ):
            raise ValueError( "pipe argument must be a list or a tuple" )
        if not isinstance( pipe_kwargs , (list,tuple) ):
            raise ValueError( "pipe_kwargs argument must be a list or a tuple" )
        if not len(pipe) == len(pipe_kwargs):
            if len(pipe_kwargs) == 0:
                pipe_kwargs = [ {} for _ in range(len(pipe)) ]
            else:
                raise ValueError( "Incoherent length between pipe and pipe_kwargs" )
        
        self._pipe = [ p(**kwargs) for p,kwargs in zip(pipe,pipe_kwargs) ]
        if bc_method is not None:
            self._bc_method  = bc_method( **bc_method_kwargs )
        else:
            self._bc_method = IdBC()
        
        self._kind   = None
        self._checkf = lambda x : True if x is None else checkf(x)
        self._check  = None
        ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """
        Transformation to apply before the bias correction method
        """
        return X
    ##}}}
    
    def itransform( self , X: _Array ) -> _Array:##{{{
        """
        Transformation to apply after the bias correction method
        """
        return X
    ##}}}
    
    def _pipe_transform( self , X: _NArray , kind: str ) -> _NArray:##{{{
        """Apply all pipe inverse transform

        Parameters
        ----------
        X: numpy.ndarray | None
            Data to apply transform
        kind: str
            If model in calibration or projection period, or reference

        Returns
        -------
        Xt: numpy.ndarray | None
            Data with transform applied
        """
        if X is None:
            return None
        Xt = X.copy()
        
        self._kind = kind
        for p in self._pipe[::-1]:
            p._kind      = kind
            p._bc_method = self._bc_method
            Xt = p.transform(Xt)
        
        Xt = self.transform(Xt)
        
        return Xt
    ##}}}
    
    def _pipe_itransform( self , Xt: _NArray , kind: str ) -> _NArray:##{{{
        """Apply all pipe inverse transform

        Parameters
        ----------
        Xt: numpy.ndarray | None
            Data to apply inverse transform
        kind: str
            If model in calibration or projection period, or reference

        Returns
        -------
        X: numpy.ndarray | None
            Data with inverse transform applied
        """
        if Xt is None:
            return None
        X = Xt.copy()
        
        self._kind = kind
        X = self.itransform(X)
        for p in self._pipe:
            p._kind      = kind
            p._bc_method = self._bc_method
            X = p.itransform(X)
        
        return X
    ##}}}
    
    def fit( self , Y0: _Array , X0: _Array , X1: _NArray = None ) -> Self:##{{{
        """Fit the bias correction method after the pre-processing.

        Parameters
        ----------
        Y0: numpy.ndarray
            Reference in calibration period
        X0: numpy.ndarray
            Biased model in calibration period
        X1: numpy.ndarray | None
            Biased model in projection period

        Returns
        -------
        Self: PrePostProcessing
            The fitted class
        """
        
        ## The check
        self._check = all([self._checkf(K) for K in [Y0,X0,X1]])
        
        if not self._check:
            raise ValueError( "PrePostProcessing: invalid check (see the 'checkf' parameter" )
        
        ## The transform
        Y0t = self._pipe_transform( Y0 , "Y0" )
        X0t = self._pipe_transform( X0 , "X0" )
        X1t = self._pipe_transform( X1 , "X1" )
        
        ## The fit
        if X1 is None:
            self._bc_method.fit( Y0t , X0t )
        else:
            self._bc_method.fit( Y0t , X0t , X1t )
        
        return self
    ##}}}
    
    def _predictZ0( self , X0: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """Predict the bias correction method after the pre-processing, then
        apply the post-processing operation.

        Parameters
        ----------
        X0: numpy.ndarray | None
            Biased model in calibration period

        Returns
        -------
        Z0: numpy.ndarray | None
            Correction of biased model in calibration period
        """
        if not self._check:
            return X0
        if X0 is None:
            return None
        
        X0t = self._pipe_transform( X0 , "X0" )
        Z0t = self._bc_method._predictZ0( X0t , **kwargs )
        Z0  = self._pipe_itransform( Z0t , "X0" )

        return Z0
    ##}}}
    
    def _predictZ1( self , X1: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """Predict the bias correction method after the pre-processing, then
        apply the post-processing operation.

        Parameters
        ----------
        X1: numpy.ndarray | None
            Biased model in projection period

        Returns
        -------
        Z1: numpy.ndarray | None
            Correction of biased model in projection period
        """
        if not self._check:
            return X1
        if X1 is None:
            return None
        
        X1t = self._pipe_transform( X1 , "X1" )
        Z1t = self._bc_method._predictZ1( X1t , **kwargs )
        Z1  = self._pipe_itransform( Z1t , "X1" )
        
        return Z1
    ##}}}
    
    def predict( self , X1: _NArray = None , X0: _NArray = None , **kwargs: Any ) -> _NArray | tuple[_NArray,_NArray]:##{{{
        """Predict the bias correction method after the pre-processing, then
        apply the post-processing operation.

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
        
        Z0 = self._predictZ0( X0 , **kwargs )
        Z1 = self._predictZ1( X1 , **kwargs )
        
        if X0 is None:
            return Z1
        elif X1 is None:
            return Z0
        else:
            return Z1,Z0
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def name(self) -> str:
        name = self._name
        if len(self._pipe) > 0:
            name = name + ":" + ":".join([ p.name for p in self._pipe ])
        name = name + f":{self._bc_method.name}"
        return name
    
    ##}}}
    
##}}}

