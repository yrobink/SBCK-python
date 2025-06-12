
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
from .__PrePostProcessing import PrePostProcessing

import scipy.spatial.distance as ssd
from ..misc.__sys import deprecated


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

class OnlyFinite(PrePostProcessing):##{{{
    """This class is used to define pre/post processing class such that rows
    which contains at least one not finite value are replace by nan in the
    final correction.
    """
    
    _is_valid: dict[str,Sequence[bool]]
    _is_all_valid: dict[str,bool]
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        """
        Arguments
        ---------
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        *kwargs:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        """
        
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name = "OnlyFinite"
        self._is_valid = {}
        self._is_all_valid = {}
        
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """
        Apply the transform
        """
        Xt = X.copy()
        
        ## Read valid
        self._is_valid[self._kind] = np.all( np.isfinite(X) , 1 )
        self._is_all_valid[self._kind] = np.all( self._is_valid[self._kind])
        
        ## If all are valid, good
        if self._is_all_valid[self._kind]:
            return Xt
        
        return Xt[self._is_valid[self._kind],:]
    ##}}}
    
    def itransform( self , Xt: _Array ) -> _Array:##{{{
        """
        Apply the inverse transform
        """
        
        ## If all are valid, good
        if self._is_all_valid[self._kind]:
            return Xt.copy()
        
        X = np.zeros( (self._is_valid[self._kind].size,Xt.shape[1]) ) + np.nan
        X[self._is_valid[self._kind],:] = Xt
        
        return X
    ##}}}
    
##}}}

class OnlyFiniteAnalog(PrePostProcessing):##{{{
    """This class is used to define pre/post processing class such that rows
    which contains at least one not finite value are replace by a row with
    finite values.
    
    The critera used is:
    - All variables in 'analog_var' must be finite,
    - The proportion of non-finite values must be lower than 'threshold'
    
    Otherwise, the SBCK.ppp.OnlyFinite strategy is used (rows are replace by nan
    in the final correction)
    
    """
    
    analog_var: _Cols
    threshold: float
    _is_valid: dict[str,Sequence[bool]]
    _is_all_valid: dict[str,bool]

    def __init__( self , analog_var: _Cols , *args: Any , threshold: float = 0.05 , **kwargs: Any ) -> None:##{{{
        """
        Arguments
        ---------
        analog_var: Sequence[int] | int | None
            Columns used to determine analogs, all values must be finite
        threshold: float
            Maximal proportion of non-finite values (between 0 and 1)
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        *kwargs:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        """
        
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name = "OnlyFiniteAnalog"
        
        self.analog_var = np.array([analog_var]).ravel()
        self.threshold  = threshold
        self._is_valid = {}
        self._is_all_valid = {}
        
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """
        Apply the transform
        """
        Xt = X.copy()
        
        pnan_var = [i for i in range(X.shape[1]) if i not in self.analog_var]
        
        ## Check validity of analog variables
        Xf = np.isfinite(X[:,self.analog_var])
        if not Xf.all():
            raise ValueError("All values of analogs variables must be finite!")
        
        ## Read valid
        self._is_valid[self._kind]     = np.all( np.isfinite(X[:,pnan_var]) , 1 )
        self._is_all_valid[self._kind] = np.all( self._is_valid[self._kind])
        
        ## If all are valid, good
        if self._is_all_valid[self._kind]:
            return Xt
        
        ## Too many invalid (> threshold)
        if np.sum(self._is_valid[self._kind]) / X.shape[0] < 1 - self.threshold:
            return Xt[self._is_valid[self._kind],:]
        
        ## OK, we replace by analogs
        D = ssd.cdist( Xt[self._is_valid[self._kind],:][:,self.analog_var] , Xt[~self._is_valid[self._kind],:][:,self.analog_var] ).argmin(0)
        Xt[~self._is_valid[self._kind],:] = Xt[self._is_valid[self._kind],:][D,:]
        self._is_all_valid[self._kind] = True
        
        return Xt
    ##}}}
    
    def itransform( self , Xt: _Array ) -> _Array:##{{{
        """
        Apply the inverse transform
        """
        
        ## If all are valid, good
        if self._is_all_valid[self._kind]:
            return Xt.copy()
        
        X = np.zeros( (self._is_valid[self._kind].size,Xt.shape[1]) ) + np.nan
        X[self._is_valid[self._kind],:] = Xt
        
        return X
    ##}}}
    
##}}}


######################
## Deprecated names ##
######################

@deprecated( "PPPRemoveNotFinite is renamed OnlyFinite since the version 2.0.0" )
class PPPRemoveNotFinite(OnlyFinite):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPRemoveNotFinite"
    ##}}}
    
##}}}

@deprecated( "PPPNotFiniteAnalog is renamed OnlyFiniteAnalog since the version 2.0.0" )
class PPPNotFiniteAnalog(OnlyFiniteAnalog):##{{{
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPNotFiniteAnalog"
    ##}}}
    
##}}}


