
## Copyright(c) 2023 / 2025 Yoann Robin
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


from .__PrePostProcessing import PrePostProcessing
from ..tools.__linalg import as2d
from ..tools.__sys import deprecated

import numpy as np


#############
## Classes ##
#############

class FilterWarnings(PrePostProcessing):##{{{
	"""
	SBCK.ppp.FilterWarnings
	=======================
	
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
		self._name = "FilterWarnings"
##}}}

class Xarray(PrePostProcessing):###{{{
	"""
	SBCK.ppp.Xarray
	===============
	
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
		self._name = "Xarray"
	
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

class As2d(PrePostProcessing):##{{{
	"""
	SBCK.ppp.As2d
	=============
	
	This PPP method is used to transform input in 2d array. All dimensions
	except the first are flatten. The predict method keep the shape.
	"""
	
	def __init__( self , *args , **kwargs ):##{{{
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
		self._name  = "As2d"
		self._shape = {}
	##}}}
	
	def transform( self , X ):##{{{
		self._shape[self._kind] = X.shape
		return as2d(X)
	##}}}
	
	def itransform( self , Xt ):##{{{
		return Xt.reshape(self._shape[self._kind])
	##}}}
	
##}}}


######################
## Deprecated names ##
######################

@deprecated( "PPPIgnoreWarnings is renamed FilterWarnings since the version 2.0.0" )
class PPPIgnoreWarnings(FilterWarnings):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPIgnoreWarnings"
	##}}}
	
##}}}

@deprecated( "PPPXarray is renamed Xarray since the version 2.0.0" )
class PPPXarray(Xarray):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPXarray"
	##}}}
	
##}}}




