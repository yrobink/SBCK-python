# -*- coding: utf-8 -*-

## Copyright(c) 2024 Yoann Robin
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

from .__decorators import io_fit
from .__decorators import io_predict


#############
## Classes ##
#############

class AbstractBC:##{{{
	"""
	SBCK.AbstractBC
	===============
	Base class of Bias Correction methods. Can be used only to be derived, or
	to check if a variable is an instance of a bias correction methods.
	"""
	
	def __init__( self , name , *args , **kwargs ):##{{{
		self._name = name
		self._ndim = 0
	##}}}
	
	## sys ##{{{
	
	def __str__(self):
		return f"SBCK.{self.name}"
	
	def __repr__(self):
		return self.__str__()
	
	##}}}
	
	## Properties ##{{{
	
	@property
	def ndim(self):
		return self._ndim
	
	@property
	def name(self):
		return self._name
	
	##}}}
	
##}}}

class MultiUBC(AbstractBC):##{{{
	
	"""
	SBCK.MultiUBC
	=============
	This class is used to transform a 1D bias correction method (as Quantile 
	mapping) to perform in a multivariate context, but margins per margins.
	"""
	
	
	def __init__( self , name , ubcm , args = None , kwargs = None ):##{{{
		"""
		SBCK.MultiUBC.__init__
		======================
		
		Arguments
		---------
		name: str
			Name of the BC method, pass to SBCK.AbstractBC
		ubcm: AbstractBC based class
			Univariate bias correction method class
		args:
			List of args for at each dimensions given at 'ubcm'. If args is not
			a list or a tuple, it is duplicated for each dimensions.
		kwargs:
			List of kwargs for at each dimensions given at 'ubcm'. If kwargs is
			not a list or a tuple, it is duplicated for each dimensions.
		"""
		super().__init__(name)
		self.ubcm_class  = ubcm
		self.ubcm        = []
		self.ubcm_args   = args
		self.ubcm_kwargs = kwargs
	##}}}
	
	def _check_ubcm_args_kwargs( self , *args ):##{{{
		
		## Check args
		if self.ubcm_args is None:
			self.ubcm_args = [ [] for _ in range(self.ndim) ]
		elif isinstance(self.ubcm_args,(list,tuple)):
			if not len(self.ubcm_args) == self.ndim:
				if len(self.ubcm_args) == 1:
					self.ubcm_args = [ self.ubcm_args[0] for _ in range(self.ndim) ]
				else:
					raise ValueError( f"Len of args must match the number of dimensions '{len(self.ubcm_args)} != {self.ndim}'" )
		else:
			self.ubcm_args = [ self.ubcm_args for _ in range(self.ndim) ]
		for arg in self.ubcm_args:
			if not isinstance(arg,(list,tuple)):
				raise ValueError( f"args must be a list of a tuple of list or tuple" )
		
		## Check kwargs
		if self.ubcm_kwargs is None:
			self.ubcm_kwargs = [ {} for _ in range(self.ndim) ]
		elif isinstance(self.ubcm_kwargs,(list,tuple)):
			if not len(self.ubcm_kwargs) == self.ndim:
				if len(self.ubcm_kwargs) == 1:
					self.ubcm_kwargs = [ self.ubcm_kwargs[0] for _ in range(self.ndim) ]
				else:
					raise ValueError( f"Len of kwargs must match the number of dimensions '{len(self.ubcm_kwargs)} != {self.ndim}'" )
		else:
			self.ubcm_kwargs = [ self.ubcm_kwargs for _ in range(self.ndim) ]
		for kwarg in self.ubcm_kwargs:
			if not isinstance(kwarg,dict):
				raise ValueError( f"kwargs must be a list of dict" )
	##}}}
	
	@io_fit
	def fit( self , *args ):##{{{
		"""
		Fit the bias correction method
		
		Parameters
		----------
		Y0	: np.ndarray
			Reference dataset during calibration period
		X0	: np.ndarray
			Biased dataset during calibration period
		X1	: np.ndarray
			Biased dataset during projection period, if the method is non-stationary
 		"""
		
		## Check kw-args of input
		self._check_ubcm_args_kwargs(*args)
		
		## Loop of fit
		for i in range(self.ndim):
			self.ubcm.append( self.ubcm_class( *self.ubcm_args[i] , **self.ubcm_kwargs[i] ).fit( *[X[:,i] for X in args] ) )
		
		return self
	##}}}
	
	@io_predict
	def predict( self , *args ):##{{{
		"""
		Predict the correction
		
		Parameters
		----------
		X1	: np.ndarray
			Biased dataset during projection period, if the method is non-stationary
		X0	: np.ndarray
			Biased dataset during calibration period, optional if the method is non-stationary
 		"""
		
		## Output
		oargs2d = [np.zeros_like(X) for X in args]
		
		## Loop of fit
		for i in range(self.ndim):
			res = self.ubcm[i].predict( *[X[:,i] for X in args] )
			if len(args) == 1:
				res = [res]
			for j in range(len(res)):
				oargs2d[j][:,i] = res[j]
		
		return tuple(oargs2d)
	##}}}
	
##}}}

