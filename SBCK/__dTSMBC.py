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
from .__dOTC import OTC
from .__dOTC import dOTC
from .__decorators import io_fit

from .stats.__shift import Shift


############
## Typing ##
############

from typing import Self
from typing import Any
_Array = np.ndarray
_NArray = _Array | None


###########
## Class ##
###########

class TSMBC(AbstractBC):##{{{
    """Time Shifted Multivariate Bias Correction.
    
    References
    ----------
    [1] Robin, Y. and Vrac, M.: Is time a variable like the others in
    multivariate statistical downscaling and bias correction?, Earth Syst.
    Dynam. Discuss. [preprint], https://doi.org/10.5194/esd-2021-12, in review,
    2021.
    
    Attributes
    ----------
    bc_method : An element of SBCK
        Bias correction method
    shift     : Shift class
        class used to shift and un-shift data
    """
    
    bc_method: AbstractBC
    shift: Shift

    def __init__( self , lag: int , bc_method: AbstractBC = OTC , method: str = "row" , ref: int | str = "middle" , **kwargs: Any ):##{{{
        """
        Parameters
        ----------
        lag       : int
            Time lag of the shift
        bc_method : A class of SBCK
            bias correction method used, default is SBCK.OTC
        method    : str
            inverse method for shift, see SBCK.tools.Shift
        ref       : int | str
            Reference columns/rows for inverse, see SBCK.tools.Shift, default is 0.5 * (lag+1)
        **kwargs  : arguments of bc_method
        
        """
        super().__init__( "TSMBC" , "S" )
        self.bc_method = bc_method(**kwargs)
        if ref == "middle": ref = int(0.5*(lag+1))
        self.shift     = Shift( lag , method , ref )
    ##}}}
    
    ## Properties {{{
    
    @property
    def ref(self) -> int:
        return self.shift.ref
    
    @ref.setter
    def ref( self , _ref: int ) -> None:
        self.shift.ref = _ref
    
    @property
    def method(self) -> str:
        return self.shift.method
    
    @method.setter
    def method( self , _method: str ) -> None:
        self.shift.method = _method
    ##}}}
    
    def fit( self , Y0: _Array , X0: _Array ) -> Self:##{{{
        """
        Parameters
        ----------
        Y0: numpy.ndarray | None
            Reference in calibration period
        X0: numpy.ndarray | None
            Biased model in calibration period
        
        Returns
        -------
        bcm: SBCK.TSMBC
            Bias Correction class fitted
        """
        Xs = self.shift.transform(X0)
        Ys = self.shift.transform(Y0)
        self.bc_method.fit( Ys , Xs )
        
        return self
    ##}}}
    
    def _predictZ0( self , X0: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """
        Parameters
        ----------
        X0: numpy.ndarray
            Biased model in calibration period
        reinfer_X0: bool
            If the law of X0 must be fitted again

        Returns
        -------
        Z0: numpy.ndarray | None
            Corrected biased model in calibration period
        """
        if X0 is None:
            return None
        Xs = self.shift.transform(X0)
        return self.shift.inverse( self.bc_method.predict( Xs , **kwargs ) )
    ##}}}
    
##}}}

class dTSMBC(AbstractBC):##{{{
    """Time Shifted Multivariate Bias Correction where observations are unknown
    
    Attributes
    ----------
    bc_method : An element of SBCK
        Bias correction method
    shift     : Shift class
        class used to shift and un-shift data
    
    References
    ----------
    [1] Robin, Y. and Vrac, M.: Is time a variable like the others in
    multivariate statistical downscaling and bias correction?, Earth Syst.
    Dynam. Discuss. [preprint], https://doi.org/10.5194/esd-2021-12, in review,
    2021.
    """
    
    bc_method: AbstractBC
    shift: Shift
    
    def __init__( self , lag: int , bc_method: AbstractBC = dOTC , method: str = "row" , ref: int | str = "middle" , **kwargs: Any ):##{{{
        """
        Parameters
        ----------
        lag       : int
            Time lag of the shift
        bc_method : An element of SBCK
            bias correction method used, default is SBCK.dOTC()
        method    : str
            inverse method for shift, see SBCK.tools.Shift
        ref       : int | str
            Reference columns/rows for inverse, see SBCK.tools.Shift, default is 0.5 * (lag+1)
        **kwargs  : arguments of bc_method
        
        """
        super().__init__( "dTSMBC" , "NS" )
        self.bc_method = bc_method(**kwargs)
        if ref == "middle": ref = int(0.5*(lag+1))
        self.shift     = Shift( lag , method , ref )
    ##}}}
    
    ## Methods and properties ##{{{
    @property
    def ref(self) -> int:
        return self.shift.ref
    
    @ref.setter
    def ref( self , _ref: int ) -> None:
        self.shift.ref = _ref
    
    @property
    def method(self) -> str:
        return self.shift.method
    
    @method.setter
    def method( self , _method: str ) -> None:
        self.shift.method = _method
    
    ##}}}
    
    @io_fit
    def fit( self , Y0: _Array , X0: _Array , X1: _Array ) -> Self:##{{{
        """
        Parameters
        ----------
        Y0: numpy.ndarray
            Reference in calibration period
        X0: numpy.ndarray
            Biased model in calibration period
        X1: numpy.ndarray
            Biased model in projection period

        Returns
        -------
        bcm: SBCK.dTSMBC
            Bias Correction class fitted
        """
        Y0s = self.shift.transform(Y0)
        X0s = self.shift.transform(X0)
        X1s = self.shift.transform(X1)
        self.bc_method.fit( Y0s , X0s , X1s )
        
        return self
    ##}}}
    
    def _predictZ0( self , X0: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """
        Parameters
        ----------
        X0: numpy.ndarray
            Biased model in calibration period

        Returns
        -------
        Z0: numpy.ndarray | None
            Corrected biased model in calibration period
        """
        if X0 is None:
            return None
        X0s = self.shift.transform(X0)
        Z0s = self.bc_method._predictZ0( X0s , **kwargs )
        Z0  = self.shift.inverse(Z0s)
        
        return Z0
    ##}}}
    
    def _predictZ1( self , X1: _NArray , **kwargs: Any ) -> _NArray:##{{{
        """
        Parameters
        ----------
        X1: numpy.ndarray
            Biased model in calibration period

        Returns
        -------
        Z1: numpy.ndarray | None
            Corrected biased model in calibration period
        """
        if X1 is None:
            return None
        X1s = self.shift.transform(X1)
        Z1s = self.bc_method._predictZ1( X1s , **kwargs )
        Z1  = self.shift.inverse(Z1s)
        return Z1
    ##}}}
    
##}}}



