
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

from .__dOTC import Univariate_dOTC1d
from .__AbstractBC import MultiUBC
from .stats.__rv_extend import rv_empirical


############
## Typing ##
############

from typing import Any
from .__AbstractBC import _rv_type
from .__AbstractBC import _mrv_type


#############
## Classes ##
#############

class Univariate_QDM(Univariate_dOTC1d):##{{{
    """Quantile Delta Mapping, from [1]

    The implementation proposes the additive QDM, equivalent to Equidistant
    CDF matching og [2]. The multiplicative form can be retrieve by using 
    the ppp SBCK.ppp.LFLog.
    
    References
    ----------
    [1] Cannon, A. J., Sobie, S. R., and Murdock, T. Q.: Bias correction of
    simulated precipitation by quantile mapping: how well do methods preserve
    relative changes in quantiles and extremes?, J. Climate, 28, 6938–6959,
    https://doi.org/10.1175/JCLI-D-14- 00754.1, 2015.
    [2] H., J. Sheffield, and E. F. Wood, 2010: Bias correction of monthly
    precipitation and temperature fields from Intergovernmental Panel on
    Climate Change AR4 models using equidistant quantile matching. J. Geophys.
    Res., 115, D10101, doi:10.1029/2009JD012882.
    """
    def __init__( self , *args: Any , rvY0: _rv_type = rv_empirical , rvX0: _rv_type = rv_empirical , rvX1: _rv_type = rv_empirical , **kwargs: Any ):##{{{
        """
        Parameters
        ----------
        rvY0: type | rv_base
            Law of references
        rvX0: type | rv_base
            Law of models in calibration period
        rvX1: type | rv_base
            Law of models in projection period
        delta: str
            delta method, 'additive' or 'multiplicative'
        """
        self._name = "QDM"
        super().__init__( *args , cfactor = 1. , **kwargs )
    ##}}}
    
##}}}

class QDM(MultiUBC):##{{{
    __doc__ = Univariate_QDM.__doc__
    
    def __init__( self , *args: Any , rvY0: _mrv_type = rv_empirical , rvX0: _mrv_type = rv_empirical , rvX1: _mrv_type = rv_empirical , **kwargs: Any ) -> None:##{{{
        
        __doc__ = Univariate_QDM.__init__.__doc__
        
        ## And init upper class
        gkwargs = { **kwargs , **{ 'rvY0' : rvY0 , 'rvX0' : rvX0 } }
        super().__init__( "QDM" , Univariate_QDM , args = args , kwargs = gkwargs )
    ##}}}
    
##}}}




