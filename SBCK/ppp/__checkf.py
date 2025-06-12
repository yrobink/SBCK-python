
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
from ..misc.__sys import deprecated


###############
## Functions ##
###############

def allfinite(X):##{{{
	"""
	SBCK.ppp.allfinite
	==================
	
	Parameters
	----------
	X:
		Input numpy array
	
	Returns
	-------
	bool:
		Return true if all values of X are finite (numpy.finite), else return
		False.
	
	"""
	return np.all(np.isfinite(X))
	##}}}

def atleastonefinite(X):##{{{
	"""
	SBCK.ppp.atleastonefinite
	=========================
	
	Parameters
	----------
	X:
		Input numpy array
	
	Returns
	-------
	bool:
		Return true if any values of X are finite (numpy.finite), else return
		False.
	
	"""
	return np.any(np.isfinite(X))
	##}}}


################
## Deprecated ##
################

@deprecated( "skipNotValid is renamed allfinite since the version 2.0.0" )
def skipNotValid(X):##{{{
	return allfinite(X)
##}}}


