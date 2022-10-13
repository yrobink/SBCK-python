
## Copyright(c) 2022 Yoann Robin
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


###########
## Class ##
###########

class BCISkipNotValid:
	
	"""
	SBCK.ppp.BCISkipNotValid
	========================
	
	Intercept a bias correction method given by 'bc_method' such that if Y0, X0
	or X1 contain a not valid value, the correction contains only np.nan values.
	
	Examples
	--------
	
	>>> Y0,X0,X1 = SBCK.datasets.like_tas_pr(2000)
	>>> X0[10,0] = np.nan
	
	>>> bcm = SBCK.ppp.BCISkipNotValid( bc_method = SBCK.CDFt ) ## bcm is the CDFt method
	>>> bcm.fit(Y0,X0,X1)
	>>> Z1,Z0 = bcm.predict(X1,X0)
	
	"""
	
	def __init__( self , bc_method , **bc_method_kwargs ):##{{{
		"""
		Initialisation of BCISkipNotValid.
		
		Parameters
		----------
		bc_method: a bias correction method of SBCK
		bc_method_kwargs: keywords arguments passed to bc_method
		
		"""
		self._bc_method = bc_method(**bc_method_kwargs)
		self._isgood = True
	##}}}
	
	def fit( self , Y0 , X0 , X1 = None ):##{{{
		"""
		Fit of CDFt model
		
		Parameters
		----------
		Y0	: np.array[ shape = (n_samples,n_features) ]
			Reference dataset during calibration period
		X0	: np.array[ shape = (n_samples,n_features) ]
			Biased dataset during calibration period
		X1	: np.array[ shape = (n_samples,n_features) ]
			Biased dataset during projection period
		"""
		
		self._isgood = np.all(np.isfinite(Y0)) and np.all(np.isfinite(X0))
		if X1 is not None:
			self._isgood = self._isgood and np.all(np.isfinite(X1))
		
		if self._isgood:
			if X1 is not None:
				self._bc_method.fit( Y0 , X0 , X1 )
			else:
				self._bc_method.fit( Y0 , X0 )
	##}}}
	
	def predict( self , X1 , X0 = None ):##{{{
		"""
		Perform the bias correction
		Return Z1 if X0 is None, else return a tuple Z1,Z0
		
		Parameters
		----------
		X1 : np.array[ shape = (n_sample,n_features) ]
			Array of value to be corrected in projection period
		X0 : np.array[ shape = (n_sample,n_features) ] or None
			Array of value to be corrected in calibration period, optional
		
		Returns
		-------
		Z1 : np.array[ shape = (n_sample,n_features) ]
			Return an array of correction in projection period
		Z0 : np.array[ shape = (n_sample,n_features) ] or None
			Return an array of correction in calibration period
		"""
		
		if self._isgood:
			if X0 is not None:
				Z = self._bc_method.predict( X1 , X0 )
			else:
				Z = self._bc_method.predict( X1 )
		else:
			Z1 = np.zeros_like(X1) + np.nan
			if X0 is not None:
				Z0 = np.zeros_like(X0) + np.nan
				Z = (Z1,Z0)
			else:
				Z = Z1
		
		return Z
	##}}}

