# -*- coding: utf-8 -*-

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

#############
## Imports ##
#############

import numpy as np
import scipy.stats as sc

from .stats.__rv_extend import rv_base
from .stats.__rv_extend import rv_scipy

from .__decorators import io_fit
from .__decorators import io_predict


############
## Typing ##
############

from typing import Self
from typing import Sequence
from typing import Any
_rv_scipy        = sc._distn_infrastructure.rv_continuous
_rv_scipy_frozen = sc._distn_infrastructure.rv_continuous_frozen
_rv_type  = type | rv_base | _rv_scipy | _rv_scipy_frozen
_mrv_type = _rv_type | Sequence[_rv_type]
_Array = np.ndarray
_NArray = _Array | None


#############
## Classes ##
#############

class AbstractBC:##{{{
    """Base class of Bias Correction methods. Can be used only to be derived,
    or to check if a variable is an instance of a bias correction methods.
    """
    
    _name: str
    _nsk: str
    _ndim: int

    def __init__( self , name: str , non_stationarity_kind: str , *args: Any , **kwargs: Any ) -> None:##{{{
        """
        Parameters
        ----------
        name: str
            Name of the BC method
        non_stationarity_kind: str
            If stationary or not
        """
        self._name = name
        self._nsk  = non_stationarity_kind
        self._ndim = 0
        
        if self._nsk not in ["S","NS","SNS","None"]:
            raise ValueError("non_stationarity_kind must be 'S', 'NS', 'SNS' or 'None'")
        
    ##}}}
    
    ## sys ##{{{
    
    def __str__(self) -> str:
        return f"SBCK.{self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def ndim(self) -> int:
        return self._ndim
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_non_stationary(self) -> bool:
        return self._nsk  == "NS"
    
    @property
    def is_stationary(self) -> bool:
        return self._nsk == "S"
    
    @property
    def stationarity_is_not_relevant(self) -> bool:
        return self._nsk == "None"
    
    ##}}}
    
    ## Predict methods ##{{{
    
    def _predictZ0( self , Z0: _NArray , **kwargs ) -> _NArray:
        """Predict fonction in calibration period"""
        raise NotImplementedError
    
    def _predictZ1( self , Z1: _NArray , **kwargs ) -> _NArray:
        """Predict fonction in projection period"""
        raise NotImplementedError
    
    def _return_predict_pair( self  , Z1: _NArray = None , Z0: _NArray = None ) -> tuple[_Array,_Array] | _NArray:
        """Predict function which choose if a tuple or an array is returned"""
        if Z0 is not None and Z1 is not None:
            return Z1,Z0
        if Z1 is None:
            return Z0
        if Z0 is None:
            return Z1

    @io_predict
    def predict( self , *args: _NArray , **kwargs: Any ) -> tuple[_Array,_Array] | _NArray:
        """Global predict function"""
        
        if self.is_stationary:
            if len(args) > 1:
                raise ValueError("Too many positional arguments, only 0 or 1 can be given")
            X0 = kwargs.get("X0")
            if len(args) == 1:
                X0 = args[0]
            Z0 = self._predictZ0( X0 , **kwargs )
            return Z0
        
        if self.is_non_stationary:
            if len(args) > 2:
                raise ValueError("Too many positional arguments, only 0, 1 or 2 can be given")
            X1 = kwargs.get("X1")
            X0 = kwargs.get("X0")
            if len(args) == 1:
                X1 = args[0]
            elif len(args) == 2:
                X1,X0 = args
            Z1 = self._predictZ1( X1 , **kwargs )
            Z0 = self._predictZ0( X0 , **kwargs )

            return self._return_predict_pair( Z1 = Z1 , Z0 = Z0 )
    ##}}}
    
##}}}

class UnivariateBC(AbstractBC):##{{{
    """Base class of univaruate Bias Correction methods. Can be used only to be
    derived, or to check if a variable is an instance of a univariate bias
    correction methods.
    """
    
    def __init__( self , name: str , non_stationarity_kind: str , *args: Any , **kwargs: Any ) -> None:##{{{
        """
        Parameters
        ----------
        name: str
            Name of the BC method
        non_stationarity_kind: str
            If stationary or not
        """
        super().__init__( name , non_stationarity_kind )
        self._ndim = 1
        
    ##}}}
    
    def _init( self , rv: _rv_type ) -> tuple[type,bool,rv_base | None]:##{{{
        """Method used to find parameters of a univariate random variable
        
        Parameters
        ----------
        rv: type | SBCK.stats.rv_base | sc._distn_infrastructure.rv_continuous | sc._distn_infrastructure.rv_continuous_frozen

        Returns
        -------
        type_: SBCK.stats.rv_base
            A rv type
        freeze_: bool
            If the law is frozen
        rv_: SBCK.stats.rv_base | None
            The law if frozen, else None
        """
        type_   = None
        freeze_ = None
        rv_     = None
        if isinstance( rv , (type,_rv_scipy) ):
            type_   = rv
            freeze_ = False
            rv_     = None
        else:
            freeze_ = True
            if isinstance( rv , _rv_scipy_frozen ):
                type_   = rv_scipy
                rv_     = rv_scipy(rv)
            else:
                type_   = type(rv)
                rv_     = rv

        return type_,freeze_,rv_
    ##}}}
    
    def _fit( self , X: _NArray , type_: type , freeze: bool , rv: rv_base | None ) -> rv_base:##{{{
        """Method used to fit univariate random variable
        
        Parameters
        ----------
        X: np.ndarray | None
            Data to fit
        type_: SBCK.stats.rv_base
            A rv type
        freeze_: bool
            If the law is frozen
        rv_: SBCK.stats.rv_base | None
            The law if frozen, else None

        Returns
        -------
        rv: type_
        """
        if X is None or freeze:
            return rv
        
        if isinstance( type_ , _rv_scipy ):
            return rv_scipy.fit( X , type_ )
        else:
            return type_.fit( X )
    ##}}}
    
##}}}

class MultiUBC(AbstractBC):##{{{
    
    """This class is used to transform a 1D bias correction method (as Quantile 
    mapping) to perform in a multivariate context, but margins per margins.
    """
  
    ubcm_class: AbstractBC
    ubcm: Sequence[AbstractBC]
    ubcm_args: Sequence[tuple[Any,...]]
    ubcm_kwargs: Sequence[dict[Any,...]]

    @staticmethod
    def _build_margs_from_args( *args: Any , **kwargs: Any ) -> tuple[tuple[Any,...],dict[Any,...]]:##{{{
        """Transform args and kwargs on a list of args and kwargs for each
        marginals
        """
        
        ## Find size
        lsizes = set()
        for arg in args:
            if isinstance( arg , (list,tuple) ):
                lsizes.add(len(arg))
            else:
                lsizes.add(1)
        for key in kwargs:
            if isinstance( kwargs[key] , (list,tuple) ):
                lsizes.add(len(kwargs[key]))
            else:
                lsizes.add(1)
        lsizes = sorted(lsizes)
        if len(lsizes) > 2 or (len(lsizes) == 2 and 1 not in lsizes):
            raise ValueError("Inconsistent arguments (multiple size)")
        if len(lsizes) == 2:
            lsizes.remove(1)
        size = lsizes[0]
        
        zargs = []
        for arg in args:
            if size == 1:
                if isinstance( arg , (list,tuple) ):
                    zargs.append(arg)
                else:
                    zargs.append([arg])
            else:
                if isinstance( arg , (list,tuple) ):
                    zargs.append(arg)
                else:
                    zargs.append([arg for _ in range(size)])
        margs = tuple([t for t in zip(*zargs)]) 
        
        zkwargs = {}
        for key in kwargs:
            elmnt = kwargs[key]
            if size == 1:
                if isinstance( elmnt , (list,tuple) ):
                    zkwargs[key] = elmnt
                else:
                    zkwargs[key] = [elmnt]
            else:
                if isinstance( elmnt , (list,tuple) ):
                    zkwargs[key] = elmnt
                else:
                    zkwargs[key] = [elmnt for _ in range(size)]
        mkwargs = [ { key : zkwargs[key][i] for key in kwargs } for i in range(size) ]
        
        if len(margs) == 0:
            margs = [ tuple() for _ in range(size) ]
        if len(mkwargs) == 0:
            mkwargs = [ {} for _ in range(size) ]

        return margs,mkwargs
    ##}}}
    
    def __init__( self , name: str , ubcm: AbstractBC , args: tuple[Any,...] | None = None , kwargs: dict[Any,...] | None = None ) -> None:##{{{
        """
        Parameters
        ----------
        name: str
            Name of the BC method, pass to SBCK.AbstractBC
        ubcm: AbstractBC based class
            Univariate bias correction method class
        args:
            List of args for at each dimensions given at 'ubcm'. If args is not
            a list or a tuple, it is duplicated for each dimensions.
        kwargs:
            List of kwargs for at each dimensions given at 'ubcm'. If kwargs is
            not a list or a tuple, it is duplicated for each dimensions.
        """
        
        margs,mkwargs = self._build_margs_from_args( *args , **kwargs )
        
        super().__init__( name , ubcm()._nsk )
        
        self.ubcm_class  = ubcm
        self.ubcm        = []
        self.ubcm_args   = margs
        self.ubcm_kwargs = mkwargs
    ##}}}
    
    def _check_ubcm_args_kwargs( self , *args: Any ) -> None:##{{{
        """Check input args and kwargs"""
        
        ## Check args
        if self.ubcm_args is None:
            self.ubcm_args = [ [] for _ in range(self.ndim) ]
        elif isinstance(self.ubcm_args,(list,tuple)):
            if not len(self.ubcm_args) == self.ndim:
                if len(self.ubcm_args) == 1:
                    self.ubcm_args = [ self.ubcm_args[0] for _ in range(self.ndim) ]
                else:
                    raise ValueError( f"Len of args must match the number of dimensions '{len(self.ubcm_args)} != {self.ndim}'" )
        else:
            self.ubcm_args = [ self.ubcm_args for _ in range(self.ndim) ]
        for arg in self.ubcm_args:
            if not isinstance(arg,(list,tuple)):
                raise ValueError( "args must be a list of a tuple of list or tuple" )
        
        ## Check kwargs
        if self.ubcm_kwargs is None:
            self.ubcm_kwargs = [ {} for _ in range(self.ndim) ]
        elif isinstance(self.ubcm_kwargs,(list,tuple)):
            if not len(self.ubcm_kwargs) == self.ndim:
                if len(self.ubcm_kwargs) == 1:
                    self.ubcm_kwargs = [ self.ubcm_kwargs[0] for _ in range(self.ndim) ]
                else:
                    raise ValueError( f"Len of kwargs must match the number of dimensions '{len(self.ubcm_kwargs)} != {self.ndim}'" )
        else:
            self.ubcm_kwargs = [ self.ubcm_kwargs for _ in range(self.ndim) ]
        for kwarg in self.ubcm_kwargs:
            if not isinstance(kwarg,dict):
                raise ValueError( "kwargs must be a list of dict" )
    ##}}}
    
    @io_fit
    def fit( self , *args: _NArray , **kwargs: Any ) -> Self:##{{{
        """Fit the bias correction method
        
        Parameters
        ----------
        args: _Array
            Any numbers of array passed for fit
         """
        
        ## Check kw-args of input
        self._check_ubcm_args_kwargs(*args)
        
        ## Loop of fit
        for i in range(self.ndim):
            self.ubcm.append( self.ubcm_class( *self.ubcm_args[i] , **self.ubcm_kwargs[i] ).fit( *[X[:,i] for X in args] , **kwargs ) )
        
        return self
    ##}}}
    
    def _predictZ0( self , X0: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """Predict in calibration period"""
        X0 = X0.reshape(-1,self.ndim)
        Z0 = np.zeros_like(X0)
        ## Loop of fit
        for i in range(self.ndim):
            Z0[:,i] = self.ubcm[i]._predictZ0( X0[:,i] , **kwargs )
        return Z0
    ##}}}
    
    def _predictZ1( self , X1: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """Predict in projection period"""
        X1 = X1.reshape(-1,self.ndim)
        Z1 = np.zeros_like(X1)
        ## Loop of fit
        for i in range(self.ndim):
            Z1[:,i] = self.ubcm[i]._predictZ1( X1[:,i] , **kwargs )
        return Z1
    ##}}}
    
    @io_predict
    def predict( self , *args: _NArray , **kwargs: Any ) -> tuple[_NArray,...]:##{{{
        """Predict the correction
        
        Parameters
        ----------
        args: _Array
            Any numbers of array passed for predict
         """
        
        ## Output
        oargs2d = [np.zeros_like(X) for X in args]
        
        ## Loop of fit
        for i in range(self.ndim):
            res = self.ubcm[i].predict( *[X[:,i] for X in args] , **kwargs )
            if len(args) == 1:
                res = [res]
            for j in range(len(res)):
                oargs2d[j][:,i] = res[j]
        
        return tuple(oargs2d)
    ##}}}
    
##}}}

