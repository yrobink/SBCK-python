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

#############
## Imports ##
#############

import numpy as np
from .tools.__linalg import as2d


################
## Decorators ##
################

def io_fit(func):##{{{
	"""
	SBCK.io_fit
	===========
	
	Decorator of the 'fit' method of bias correction method. It is used to
	transform the input array to 2-d array (or None if None).
	
	"""
	def wrapper( self , *iargs , **kwargs ):
		
		## Pre
		#* Transform data in 2d
		args2d = [as2d(X) for X in iargs]
		
		#* Number of dimensions
		ndims = set([ X.shape[1] for X in args2d if X is not None ])
		if len(ndims) > 1:
			raise ValueError( f"Different numbers of dimensions in input: {ndims}" )
		
		self._ndim = ndims.pop()
		
		iargs2d = []
		for X in args2d:
			if X is not None:
				iargs2d.append(X)
			else:
				iargs2d.append(np.zeros((1,self._ndim)) + np.nan)
		
		## Exec
		return func( self , *iargs2d , **kwargs )
	
	return wrapper
##}}}

def io_predict(func):##{{{
	"""
	SBCK.io_predict
	===============
	
	Decorator of the 'predict' method of bias correction method. It is used to
	transform the input array to 2-d array (or None if None), and go back to
	the original shape after the correction.
	
	"""
	
	def wrapper( self , *iargs , **kwargs ):
		
		## Pre
		#* Transform data in 2d
		iargs2d = [as2d(X) for X in iargs]
		
		#* Number of dimensions
		ndims = set([ X.shape[1] for X in iargs2d])
		if len(ndims) > 1:
			raise ValueError( f"Different numbers of dimensions in input: {ndims}" )
		
		ndim = ndims.pop()
		if not self._ndim == ndim:
			raise ValueError( f"Incoherent numbers of dimensions between fitted distributions and data to correct '{self._ndim} != {ndim}" )
		
		## Exec
		oargs2d = func( self , *iargs2d , **kwargs )
		
		if isinstance(oargs2d,np.ndarray):
			oargs2d = [oargs2d]
		
		## Post
		oargs = [ Z.reshape(X.shape) for X,Z in zip(iargs,oargs2d) ]
		
		if len(oargs) == 1:
			return oargs[0]
		
		return oargs
		
	
	return wrapper
##}}}


