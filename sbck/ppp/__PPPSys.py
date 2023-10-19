
## Copyright(c) 2023 Yoann Robin
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

import warnings
from .__PrePostProcessing import PrePostProcessing

import numpy as np


#############
## Classes ##
#############

class PPPIgnoreWarnings(PrePostProcessing):##{{{
	"""
	SBCK.ppp.PPPIgnoreWarnings
	==========================
	
	This PPP method is used to supress all warnings raised by python during
	the execution
	"""
	
	def __init__( self , *args , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		warnings.simplefilter("ignore")
		PrePostProcessing.__init__( self , *args , **kwargs )
##}}}

class PPPXarray(PrePostProcessing):###{{{
	"""
	SBCK.ppp.PPPXarray
	==========================
	
	This PPP method is used to deal with xarray. The xarray interface is removed
	before the fit, and applied to output of predict method.
	"""
	
	def __init__( self , *args , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		PrePostProcessing.__init__( self , *args , **kwargs )
	
	def transform( self , X ):
		
		if self._kind == 'X0':
			self._x0 = X.copy() + np.nan
		if self._kind == 'X1':
			self._x1 = X.copy() + np.nan
		
		Xt = X.values
		if Xt.ndim == 1:
			Xt = Xt.reshape(-1,1)
		return Xt
	
	def itransform( self , Xt ):
		
		if self._kind == "X0":
			X = self._x0.copy()
			X[:] = Xt.reshape(X.shape)
		elif self._kind == "X1":
			X = self._x1.copy()
			X[:] = Xt.reshape(X.shape)
		else:
			X = Xt
		
		return X
##}}}

