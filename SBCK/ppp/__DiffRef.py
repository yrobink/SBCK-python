
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

class PreserveOrder(PrePostProcessing):##{{{
    """The inverse transform of this PPP sort the data of the column 'cols'
    along rows. It is useful for example when tas, tasmin and tasmax are
    corrected to ensure their order.
    """
    
    _cols: _Cols
    
    def __init__( self , *args: Any , cols: _Cols = None , **kwargs: Any ) -> None:##{{{
        """
        Parameters
        ----------
        args: Any
            arguments passed to super class
        cols: Sequence[int] | int | None
            Columns to apply the ppp
        kwargs: Any
            keywords arguments passed to super class
        """
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name = "PreserveOrder"
        
        self._cols = cols
        if cols is not None:
            self._cols = np.array( [cols] , dtype = int ).squeeze()
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """Apply the PreserveOrder transform, in fact just identity"""
        return X
    ##}}}
    
    def itransform( self , Xt: _Array ) -> _Array:##{{{
        """Apply the inverse PreserveOrder transform, i.e. sort along cols"""
        
        if self._cols is None:
            return Xt
        
        X = Xt.copy()
        X[:,self._cols] = np.sort( X[:,self._cols] , axis = 1 )
        
        return X
    ##}}}
    
##}}}

class DeltaRef(PrePostProcessing): ##{{{
    """Transform a dataset such that all `lower` dimensions are replaced by
    the `ref` dimension minus the `lower`; and all `upper` dimensions are
    replaced by `upper` minus `ref`.
    
    >>> ## Start with data
    >>> X    = np.random.normal(size = size).reshape(-1,1)
    >>> sign = np.random.choice( [-1,1] , nfeat - 1 , replace = True )
    >>> for s in sign:
    >>>     X = np.concatenate( (X,X[:,0].reshape(-1,1) + s * np.abs(np.random.normal( size = (size,1) ))) , -1 )
    >>> 
    >>> ## Define the PPP method
    >>> ref   = 0
    >>> lower = np.argwhere( sign == -1 ).ravel() + 1
    >>> upper = np.argwhere( sign ==  1 ).ravel() + 1
    >>> pppdr = SBCK.ppp.DeltaRef( ref , lower , upper )
    >>> 
    >>> ## And now change the dimension, and reverse the operation
    >>> Xt  = pppdr.transform(X)
    >>> Xit = pppdr.itransform(Xt)
    >>> 
    >>> print( np.max( np.abs( X - Xit ) ) ) ## == 0
    """
    
    ref: int
    lower: Sequence[int] | None
    upper: Sequence[int] | None

    def __init__( self , ref: int , *args: Any , lower: Sequence[int] | None = None , upper: Sequence[int] | None = None , **kwargs: Any ) -> None: ##{{{
        """
        Arguments
        ---------
        ref: int
            The reference dimension
        lower: Sequence[int] | None
            Dimensions lower than ref
        upper: Sequence[int] | None
            Dimensions upper than ref
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        *kwargs:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        """
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name = "DeltaRef"
        
        self.ref   = ref
        self.lower = lower
        self.upper = upper
        
        if lower is not None and len(lower) == 0:
            self.lower = None
        if upper is not None and len(upper) == 0:
            self.upper = None
        
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """
        Apply the DeltaReff transform.
        """
        
        Xt = X.copy()
        
        if self.lower is not None:
            for i in self.lower:
                Xt[:,i] = X[:,self.ref] - X[:,i]
        
        if self.upper is not None:
            for i in self.upper:
                Xt[:,i] = X[:,i] - X[:,self.ref]
        
        
        return Xt
    ##}}}
    
    def itransform( self , Xt: _Array ) -> _Array:##{{{
        """
        Apply the inverse DeltaRef transform.
        """
        
        X = Xt.copy()
        
        if self.lower is not None:
            for i in self.lower:
                X[:,i] = Xt[:,self.ref] - Xt[:,i]
        
        if self.upper is not None:
            for i in self.upper:
                X[:,i] = Xt[:,i] + Xt[:,self.ref]
        
        return X
        
        ##}}}
    
##}}}

class DeltaVars(PrePostProcessing):##{{{
    """Similar to SBCK.ppp.DeltaRef, but diff columns are replaced by diff
    column minus ref columns (or reverse if sign = -1), i.e.:
    Xt[:,diff] = sign * (X[:,diff] - X[:,ref])
    
    """
    
    ref: Sequence[int] | int
    diff: Sequence[int] | int
    sign: int

    def __init__( self , ref: Sequence[int] | int , diff: Sequence[int] | int , *args: Any , sign: int = 1 , **kwargs: Any ) -> None:##{{{
        """
        Arguments
        ---------
        ref: Sequence[int] | int 
            The reference dimensions
        diff: Sequence[int] | int 
            The difference dimensions
        sign: int
            The if upper or lower
        *args:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        *kwargs:
            All others arguments are passed to SBCK.ppp.PrePostProcessing
        """
        
        PrePostProcessing.__init__( self , *args , **kwargs )
        self._name = "DeltaVars"
        
        self.ref  = ref
        self.diff = diff
        self.sign = 1 if sign > 0 else -1
        
    ##}}}
    
    def transform( self , X: _Array ) -> _Array:##{{{
        """
        Apply the transform
        """
        Xt = X.copy()
        
        Xt[:,self.diff] = self.sign * (X[:,self.diff] - X[:,self.ref])
        
        return Xt
    ##}}}
    
    def itransform( self , Xt: _Array ) -> _Array:##{{{
        """
        Apply the inverse transform
        """
        X = Xt.copy()
        
        X[:,self.diff] = X[:,self.ref] + self.sign * X[:,self.diff]
        
        return X
    ##}}}
    
##}}}


################
## Deprecated ##
################

@deprecated( "PPPPreserveOrder is renamed PreserveOrder since the version 2.0.0" )
class PPPPreserveOrder(PreserveOrder):##{{{
    """See SBCK.ppp.PreserveOrder"""
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPPreserveOrder"
    ##}}}
    
##}}}

@deprecated( "PPPDiffRef is renamed DeltaRef since the version 2.0.0" )
class PPPDiffRef(DeltaRef):##{{{
    """See SBCK.ppp.DiffRef"""
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPDiffRef"
    ##}}}
    
##}}}

@deprecated( "PPPDiffColumns is renamed DeltaVars since the version 2.0.0" )
class PPPDiffColumns(DeltaVars):##{{{
    """See SBCK.ppp.DeltaVars"""
    
    def __init__( self , *args: Any , **kwargs: Any ) -> None:##{{{
        super().__init__( *args , **kwargs )
        self._name = "PPPDiffColumns"
    ##}}}
    
##}}}

