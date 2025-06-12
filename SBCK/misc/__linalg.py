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

###############
## Libraries ##
###############

import numpy as np


############
## Typing ##
############

_Array = np.ndarray


###############
## Functions ##
###############

def as2d( X: _Array | None ) -> _Array | None :##{{{
    """Change the shape of X to have only 2 dimensions.
    - If X.ndim == 1: Xs = X.reshape(-1,1)
    - If X.ndim > 2: Xs = X.reshape(X.shape[0],-1)
    
    Note that if X is None, as2d return also None
    
    Arguments
    ---------
    X: numpy.ndarray | None
        An array
    
    Returns
    -------
    Xs: numpy.ndarray | None
        The X such that X.ndim == 2
    """
    
    if X is None:
        return None
    
    if X.ndim == 1:
        return X.reshape(-1,1)
    
    return X.reshape(X.shape[0],-1)
##}}}

def sqrtm( M: _Array , method: str = "svd" ) -> _Array:##{{{
    """Compute the square root matrix of M, i.e. the S matrix such
    that S @ S == M.
    
    Arguments
    ---------
    M: numpy.ndarray
        A matrix
    method: str 
        can be 'svd' (use singular values decomposition, more general and
        stable) or 'eig' (use eigenvalues)
    Returns
    -------
    S: numpy.ndarray
        The square root matrix
    """
    
    match method.lower():
        case "svd":
            U,s,V = np.linalg.svd(M)
        case "eig":
            s,U = np.linalg.eig(M)
            V   = U.T
        case _:
            raise ValueError("Method must be svd or eig")
    s = np.diag(np.sqrt(s))
    S = U @ s @ V
    
    return S
##}}}

