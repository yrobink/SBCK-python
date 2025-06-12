# -*- coding: utf-8 -*-

## Copyright(c) 2024 / 2025 Yoann Robin
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

from .__stats_cpp import SparseHist as SparseHistCPP


############
## Typing ##
############

from typing import Sequence
from typing import Any

_Array = np.ndarray


###############
## Functions ##
###############

def bin_width_estimator( *args: _Array , method: str = "auto" ) -> _Array:##{{{
    """Estimate the width of the bin to build an histogram of X
    
    Parameters
    ----------
    *args: numpy.ndarray
        A dataset or a list of dataset X containing
    method : string = [ "auto" , "Sturges" , "FD" ]
        Method to estimate bin_width. If method == "auto", "Sturges" is selected if n_samples < 1000, else "FD"
    
    Returns
    -------
    bin_width : numpy.ndarray
        bin_width of each features.
    """
    if len(args) > 1:
        return np.min( [ bin_width_estimator( X , method = method ) for X in args ] , axis = 0 )
    X = args[0]
    
    if X.ndim == 1:
        X = X.reshape(-1,1)
    
    if method == "auto":
        method = "Sturges" if X.shape[0] < 1000 else "FD"
    
    match method:
        case "Sturges":
            nh = np.log2( X.shape[0] ) + 1.
            bin_width = np.zeros(X.shape[1]) + 1. / nh
        case "FD":
            bin_width = 2. * ( np.percentile( X , q = 75 , axis = 0 ) - np.percentile( X , q = 25 , axis = 0 ) ) / np.power( X.shape[0] , 1. / 3. )
        case _:
            raise ValueError("Available methods are 'auto', 'Sturges' or 'FD'")
    
    return bin_width
##}}}

#############
## Classes ##
#############

class BaseHist:##{{{
    """Basic histogram, with just center and probability.
    
    Attributes
    ----------
    - 'c': center of bins,
    - 'p': probability of bins,
    
    """
    
    _p: _Array
    _c: _Array

    def __init__( self , c: _Array , p: _Array ) -> None:##{{{
        self._p = p
        self._c = c
        if not self.shape[0] == self.sizep:
            raise ValueError("Inconsistent dimensions between center c and probabilities p")
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def c(self) -> _Array:
        return self._c
    
    @property
    def p(self) -> _Array:
        return self._p
    
    @property
    def sizep(self) -> int:
        return self._p.size
    
    @property
    def shape(self) -> tuple[int,...]:
        return self._c.shape
    
    @property
    def ndim(self) -> int:
        return self._c.ndim
    
    @property
    def size(self) -> int:
        return np.prod(self.shape)

    ##}}}
    
##}}}

class SparseHist:##{{{
    """Sparse Histogram class, interface to a c++ class.
    
    Attributes:
    - 'c': center of bins,
    - 'p': probability of bins,
    - 'shape': shape of c,
    - 'ndim': numbers of dimensions,
    - 'size': total size (product of shape),
    - 'sizep': number of bins
    """
    
    _sparse_hist: SparseHistCPP

    def __init__( self , X: _Array , bin_width: Sequence[float] | float | None = None , bin_origin: Sequence[float] | float = 0 ) -> None:##{{{
        """
        Arguments
        ---------
        
        X: numpy.ndarray
            Data to infer the histogram
        bin_width: Sequence[float] | float | None
            Width of a bin
        bin_origin: Sequence[float] | float
            left corner of one bin, default is 0
        
        """
        
        ## Check X
        if X.ndim == 1:
            X = X.reshape(-1,1)
        ndim = X.shape[1]
    
        ## Check bin_width and bin_origin
        if bin_width is None:
            bin_width = bin_width_estimator(X)
        elif np.isscalar(bin_width):
            bin_width = [bin_width for _ in range(ndim)]
        if bin_origin is None:
            bin_origin = np.zeros_like(bin_width)
        elif np.isscalar(bin_origin):
            bin_origin = [bin_origin for _ in range(ndim)]

        bin_width  = np.array(bin_width).ravel()
        bin_origin = np.array(bin_origin).ravel()
        

        ## And init the SparseHist
        self._sparse_hist = SparseHistCPP( X , bin_width , bin_origin )
    ##}}}
    
    def argwhere( self , X: np.ndarray ) -> np.ndarray:##{{{
        """Return the index of the bins in 'c' of the elements of X. So:
        x[:,i] is the bin defined by c[I[i],:]
        
        Arguments
        ---------
        X: numpy.ndarray
            Data to infer the histogram
        
        Returns
        -------
        I: np.ndarray
            Index
        
        """
        
        ## Check X
        if X.ndim == 1:
            X = X.reshape(-1,1)
        size,ndim = X.shape
        
        ## Finite index
        v    = np.isfinite(X).all(1)
        
        ## Output
        I    = np.zeros(X.shape[0]) + self.shape[0] + 1
        I[v] = self._sparse_hist.argwhere(X[v,:])
        I    = np.where( I > -1 , I , self.shape[0] )
        
        return I.astype(int)
    ##}}}
    
    ## Properties ##{{{
    
    @property
    def c(self) -> _Array:
        return self._sparse_hist.c
    
    @property
    def p(self) -> _Array:
        return self._sparse_hist.p
    
    @property
    def sizep(self) -> int:
        return self._sparse_hist.p.size
    
    @property
    def shape(self) -> tuple[int,...]:
        return self._sparse_hist.c.shape
    
    @property
    def ndim(self):
        return self._sparse_hist.c.ndim
    
    @property
    def size(self) -> int:
        return np.prod(self.shape)
    
    @property
    def bin_width(self) -> _Array:
        return self._sparse_hist.bin_width
    
    @property
    def bin_origin(self) -> _Array:
        return self._sparse_hist.bin_origin

    ##}}}
    
##}}}

