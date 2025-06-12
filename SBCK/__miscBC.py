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
from .__AbstractBC import AbstractBC
from .__decorators import io_fit
from .__decorators import io_predict


############
## Typing ##
############

from typing import Any
from typing import Self
_Array  = np.ndarray
_NArray = _Array | None


###########
## Class ##
###########

class IdBC(AbstractBC):##{{{
    """
    Identity Bias Correction. Always return X0 / X1 without use Y0.
    """
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( "IdBC" , "None" )
    ##}}}
    
    @io_fit
    def fit( self , *args: Any ) -> Self:##{{{
        """
        Parameters
        ----------
        args: Any
            Fit nothing, so any parameters can be given
        """
        
        return self
    ##}}}
    
    def _predictZ0( self , X0: _NArray , **kwargs: Any ) -> _NArray:##{{{
        return X0
    ##}}}
    
    def _predictZ1( self , X1: _NArray , **kwargs: Any ) -> _NArray:##{{{
        return X1
    ##}}}
    
    @io_predict
    def predict( self , *args: _NArray , **kwargs: Any ) -> tuple[_NArray,...]:##{{{
        return args
    ##}}}
    
##}}}

class RBC(AbstractBC):##{{{
    """Random Bias Correction. This method correct randomly X0/X1 with respect to
    Y0. Used to test if a BC is an improvement.  The fit method can be used in
    stationary or non stationary case, but in fact X0 and X1 are not used. We
    just draw uniformly values from Y0
    
    """
    
    _Y: _NArray

    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( "RBC" , "None" )
        self._Y = None
    ##}}}
    
    @io_fit
    def fit( self , Y0: _Array , *args: _NArray ) -> Self:##{{{
        """
        Fit the RBC
        
        Parameters
        ----------
        Y0    : numpy.ndarray
            Reference dataset during calibration period
        *args: numpy.ndarray | None
            Any dataset
        """
        self._Y = Y0
        
        return self
    ##}}}
    
    
    def _predictZ0( self , X0: _NArray , **kwargs: Any ) -> _NArray:##{{{
        if X0 is None:
            return None
        return self._Y[np.random.choice( self._Y.shape[0] , X0.shape[0] ),:]
    ##}}}
    
    def _predictZ1( self , X1: _NArray , **kwargs: Any ) -> _NArray:##{{{
        if X1 is None:
            return None
        return self._Y[np.random.choice( self._Y.shape[0] , X1.shape[0] ),:]
    ##}}}
    
    @io_predict
    def predict( self , X1:_NArray = None , X0: _NArray = None , **kwargs: Any ) -> _NArray | tuple[_NArray,_NArray]:##{{{
        """
        Perform the bias correction
        
        Parameters
        ----------
        X1  : numpy.ndarray | None
            Array of value to be corrected in projection period
        X0  : numpy.ndarray | None
            Array of value to be corrected in calibration period
        
        Returns
        -------
        Z1  : numpy.ndarray | None
            Return an array of correction in projection period
        Z0  : numpy.ndarray | None
            Return an array of correction in calibration period, or None
        """
        
        Z0 = self._predictZ0( X0 , **kwargs )
        Z1 = self._predictZ1( X1 , **kwargs )
        
        return self._return_predict_pair( Z1 = Z1 , Z0 = Z0 )
        
    ##}}}
    
##}}}

