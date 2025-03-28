
## Copyright(c) 2025 Yoann Robin
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

from ..__AbstractBC import AbstractBC


#############
## Classes ##
#############

class EnsITE(AbstractBC):##{{{
	"""
	EnsITE: Ensemble Independent Transfer Estimation
	================================================
	
	Main class for independent ensemble bias correction. It is equivalent to correct each members independently.
	
	The expected shape of input is:
	Y0: (time,ndim)
	X0: (run,time,ndim)
	X1: (run,time,ndim)
	
	And the numbers of run of X0 and X1 must be equal.
	
	"""
	
	def __init__( self , bcm , bcm_kwargs = {} ):##{{{
		self._bcmc        = bcm
		self._bcmc_kwargs = bcm_kwargs
		self._bcm         = []
		super().__init__( f"EnsITE:{bcm(**bcm_kwargs).name}" , "None" )
	##}}}
	
	def _reshapeX( self , X ):##{{{
		if X.ndim == 1 and self.ndim == 1:
			X = X.reshape(1,-1,1)
		if not X.ndim == 3:
			raise ValueError("Impossible to find how to reshape X0 or X1 with 3 dimensions")
		if not self.ndim == X.shape[2]:
			raise ValueError("Incoherent numbers of components between Y0 and X0 or X1")
		return X
	##}}}
	
	def fit( self , Y0 , X0 , X1 ):##{{{
		
		## Start by infer dimension from Y0
		if Y0.ndim == 1:
			Y0 = Y0.reshape(-1,1)
		if Y0.ndim > 2:
			raise ValueError("Dimension of Y0 > 2")
		self._ndim = Y0.shape[1]

		## Reshape X0 and X1
		X0 = self._reshapeX(X0)
		X1 = self._reshapeX(X1)
		
		if not X0.shape[0] == X1.shape[0]:
			raise ValueError("Different number of members between X0 and X1")
		
		## Now it is ready to fit
		for i in range(X0.shape[0]):
			self._bcm.append( self._bcmc(**self._bcmc_kwargs).fit( Y0 , X0[i,:,:] , X1[i,:,:] ) )
		
		return self
	##}}}
	
	def predict( self , X1 , X0 = None ):##{{{
		## Reshape X0 and X1
		X0 = self._reshapeX(X0)
		X1 = self._reshapeX(X1)
		
		if not X0.shape[0] == X1.shape[0]:
			raise ValueError("Different number of members between X0 and X1")
		
		## Correction
		Z1 = np.zeros_like(X1) + np.nan
		for i in range(X1.shape[0]):
			Z1[i,:,:] = self._bcm[i]._predictZ1( X1[i,:,:] )
		Z1 = Z1.reshape(X1.shape)
		
		if X0 is None:
			return Z1
		
		Z0 = np.zeros_like(X0) + np.nan
		for i in range(X0.shape[0]):
			Z0[i,:,:] = self._bcm[i]._predictZ0( X0[i,:,:] )
		Z0 = Z0.reshape(X0.shape)
		
		return Z1,Z0
	##}}}
	
##}}}

class EnsCTE(AbstractBC):##{{{
	"""
	EnsCTE: Ensemble Calibration Transfer Estimation
	================================================
	
	Main class for ensemble bias correction as defined by Vaittinada A. et al (2021) (doi:10.1038/s41598-021-82715-1).
	
	The expected shape of input is:
	Y0: (time,ndim)
	X0: (run,time,ndim)
	X1: (run,time,ndim)
	
	And the numbers of run of X0 and X1 must be equal.
	"""
	
	def __init__( self , bcm , bcm_kwargs = {} ):##{{{
		self._bcmc        = bcm
		self._bcmc_kwargs = bcm_kwargs
		self._bcm         = []
		super().__init__( f"EnsCTE:{bcm(**bcm_kwargs).name}" , "None" )
	##}}}
	
	def _reshapeX( self , X ):##{{{
		if X.ndim == 1 and self.ndim == 1:
			X = X.reshape(1,-1,1)
		if not X.ndim == 3:
			raise ValueError("Impossible to find how to reshape X0 or X1 with 3 dimensions")
		if not self.ndim == X.shape[2]:
			raise ValueError("Incoherent numbers of components between Y0 and X0 or X1")
		return X
	##}}}
	
	def fit( self , Y0 , X0 , X1 ):##{{{
		
		## Start by infer dimension from Y0
		if Y0.ndim == 1:
			Y0 = Y0.reshape(-1,1)
		if Y0.ndim > 2:
			raise ValueError("Dimension of Y0 > 2")
		self._ndim = Y0.shape[1]

		## Reshape X0 and X1
		X0 = self._reshapeX(X0)
		X1 = self._reshapeX(X1)
		
		if not X0.shape[0] == X1.shape[0]:
			raise ValueError("Different number of members between X0 and X1")
		
		## Now it is ready to fit
		for i in range(X0.shape[0]):
			self._bcm.append( self._bcmc(**self._bcmc_kwargs).fit( Y0 , X0.reshape(-1,self.ndim) , X1[i,:,:] ) )
		
		return self
	##}}}
	
	def predict( self , X1 , X0 = None ):##{{{
		
		## Reshape X0 and X1
		X0 = self._reshapeX(X0)
		X1 = self._reshapeX(X1)
		
		if not X0.shape[0] == X1.shape[0]:
			raise ValueError("Different number of members between X0 and X1")
		
		## Correction
		Z1 = np.zeros_like(X1) + np.nan
		for i in range(X1.shape[0]):
			Z1[i,:,:] = self._bcm[i]._predictZ1( X1[i,:,:] , reinfer_X1 = False )
		Z1 = Z1.reshape(X1.shape)
		
		if X0 is None:
			return Z1
		
		Z0 = np.zeros_like(X0) + np.nan
		for i in range(X0.shape[0]):
			Z0[i,:,:] = self._bcm[i]._predictZ0( X0[i,:,:] , reinfer_X0 = False )
		Z0 = Z0.reshape(X0.shape)
		
		return Z1,Z0
	##}}}
	
##}}}

