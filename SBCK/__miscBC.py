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
from .__decorators import io_fit
from .__decorators import io_predict


###########
## Class ##
###########

class IdBC(AbstractBC):##{{{
	"""
	SBCK.IdBC
	=========
	
	Description
	-----------
	Identity Bias Correction. Always return X0 / X1 without use Y0.
	
	"""
	
	def __init__( self , **kwargs ):##{{{
		"""
		Initialisation of IdBC
		
		Parameters
		----------
		
		Attributes
		----------
		"""
		super().__init__( "IdBC" , "SNS" )
	##}}}
	
	@io_fit
	def fit( self , Y0 , X0 , X1 = None ):##{{{
		"""
		Fit the RBC
		
		Parameters
		----------
		Y0	: np.ndarray
			Reference dataset during calibration period
		X0	: np.ndarray
			Biased dataset during calibration period
		X1	: np.ndarray or None
			Biased dataset during projection period. Can be None to use as a stationary bias correction method
		"""
		
		return self
	##}}}
	
	def _predictZ0( self , X0 , **kwargs ):##{{{
		return X0
	##}}}
	
	def _predictZ1( self , X1 , **kwargs ):##{{{
		return X1
	##}}}
	
	@io_predict
	def predict( self , X1 = None , X0 = None , **kwargs ):##{{{
		"""
		Perform the bias correction
		
		Parameters
		----------
		X1  : np.ndarray
			Array of value to be corrected in projection period
		X0  : np.ndarray or None
			Array of value to be corrected in calibration period
		
		Returns
		-------
		if:
			- X0 is None and X1 is not None : return Z1 = X1
			- X0 is not None and X1 is None : return Z0 = X0
			- X0 is not None and X1 is not None : return Z1,Z0 = X1,X0
		Z1 : np.ndarray
			Return an array of correction in projection period
		Z0 : np.ndarray or None
			Return an array of correction in calibration period, or None
		"""
		
		Z0 = X0
		Z1 = X1
		
		if X0 is not None and X1 is not None:
			return Z1,Z0
		if X1 is None:
			return Z0
		if X0 is None:
			return Z1
		
	##}}}
	
##}}}

class RBC(AbstractBC):##{{{
	"""
	SBCK.RBC
	========
	
	Description
	-----------
	Random Bias Correction. This method correct randomly X0/X1 with respect to
	Y0. Used to test if a BC is an improvement.  The fit method can be used in
	stationary or non stationary case, but in fact X0 and X1 are not used. We
	just draw uniformly values from Y0
	
	"""
	
	def __init__( self , **kwargs ):##{{{
		"""
		Initialisation of RBC
		
		Parameters
		----------
		
		Attributes
		----------
		"""
		super().__init__( "RBC" , "SNS" )
		self._Y = None
	##}}}
	
	@io_fit
	def fit( self , Y0 , X0 , X1 = None ):##{{{
		"""
		Fit the RBC
		
		Parameters
		----------
		Y0	: np.ndarray
			Reference dataset during calibration period
		X0	: np.ndarray
			Biased dataset during calibration period
		X1	: np.ndarray or None
			Biased dataset during projection period. Can be None to use as a stationary bias correction method
		"""
		self._Y = Y0
		
		return self
	##}}}
	
	
	def _predictZ0( self , X0 , **kwargs ):##{{{
		if X0 is None:
			return None
		return self._Y[np.random.choice( self._Y.shape[0] , X0.shape[0] ),:]
	##}}}
	
	def _predictZ1( self , X1 , **kwargs ):##{{{
		if X1 is None:
			return None
		return self._Y[np.random.choice( self._Y.shape[0] , X1.shape[0] ),:]
	##}}}
	
	
	@io_predict
	def predict( self , X1 = None , X0 = None , **kwargs ):##{{{
		"""
		Perform the bias correction
		
		Parameters
		----------
		X1  : np.ndarray
			Array of value to be corrected in projection period
		X0  : np.ndarray or None
			Array of value to be corrected in calibration period
		
		Returns
		-------
		if:
			- X0 is None and X1 is not None : return Z1
			- X0 is not None and X1 is None : return Z0
			- X0 is not None and X1 is not None : return Z1,Z0
		Z1 : np.ndarray
			Return an array of correction in projection period
		Z0 : np.ndarray or None
			Return an array of correction in calibration period, or None
		"""
		
		Z0 = self._predictZ0( X0 , **kwargs )
		Z1 = self._predictZ1( X1 , **kwargs )
		
		if X0 is not None and X1 is not None:
			return Z1,Z0
		if X1 is None:
			return Z0
		if X0 is None:
			return Z1
		
	##}}}
	
##}}}

