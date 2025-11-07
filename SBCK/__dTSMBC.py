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
from .__dOTC import OTC
from .__dOTC import dOTC

from .ppp.__Shift import Shift


############
## Typing ##
############

from typing import Any
_Array = np.ndarray
_NArray = _Array | None


###########
## Class ##
###########

class TSMBC(Shift):##{{{
    """Time Shifted Multivariate Bias Correction.
    
    References
    ----------
    [1] Robin, Y. and Vrac, M.: Is time a variable like the others in
    multivariate statistical downscaling and bias correction?, Earth Syst.
    Dynam. Discuss. [preprint], https://doi.org/10.5194/esd-2021-12, in review,
    2021.
    """

    def __init__( self , lag: int , method: str = "row" , ref: int | str = "middle" , **kwargs: Any ) -> None:##{{{
        """
        Parameters
        ----------
        lag       : int
            Time lag of the shift
        method    : str
            inverse method for shift, see SBCK.tools.Shift
        ref       : int | str
            Reference columns/rows for inverse, see SBCK.tools.Shift, default is 0.5 * (lag+1)
        **kwargs  : arguments passed to OTC
        """
        super().__init__( bc_method = OTC , bc_method_kwargs = kwargs , lag = lag , method = method , ref = ref )
    ##}}}
    
##}}}

class dTSMBC(Shift):##{{{
    """dynamical Time Shifted Multivariate Bias Correction.
    
    References
    ----------
    [1] Robin, Y. and Vrac, M.: Is time a variable like the others in
    multivariate statistical downscaling and bias correction?, Earth Syst.
    Dynam. Discuss. [preprint], https://doi.org/10.5194/esd-2021-12, in review,
    2021.
    """

    def __init__( self , lag: int , method: str = "row" , ref: int | str = "middle" , **kwargs: Any ) -> None:##{{{
        """
        Parameters
        ----------
        lag       : int
            Time lag of the shift
        method    : str
            inverse method for shift, see SBCK.tools.Shift
        ref       : int | str
            Reference columns/rows for inverse, see SBCK.tools.Shift, default is 0.5 * (lag+1)
        **kwargs  : arguments passed to OTC
        """
        super().__init__( bc_method = dOTC , bc_method_kwargs = kwargs , lag = lag , method = method , ref = ref )
    ##}}}
    
##}}}


