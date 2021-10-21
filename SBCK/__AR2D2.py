# -*- coding: utf-8 -*-

## Copyright(c) 2021 Yoann Robin
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

from .__CDFt import CDFt
from .tools.__shuffle import MVQuantilesShuffle


###########
## Class ##
###########

class AR2D2:
	"""
	SBCK.AR2D2
	==========
	
	Description
	-----------
	Multivariate bias correction with quantiles shuffle, see [1].
	
	References
	----------
	[1] Vrac, M. et S. Thao (2020). “R2 D2 v2.0 : accounting for temporal
		dependences in multivariate bias correction via analogue rank
		resampling”. In : Geosci. Model Dev. 13.11, p. 5367-5387.
		doi :10.5194/gmd-13-5367-2020.
	
	"""
	
	def __init__( self , col_cond = [1] , lag_search = 1 , lag_keep = 1 , bc_method = CDFt , **bckwargs ):##{{{
		"""
		Initialisation of AR2D2.
		
		Parameters
		----------
		col_cond : list[int]
			Conditioning columns
		lag_search: int
			Number of lags to transform the dependence structure
		lag_keep: int
			Number of lags to keep
		bc_method: SBCK.<bc_method>
			Bias correction method
		**bckwargs: ...
			all others named arguments are passed to bc_method
		"""
		self.mvq = MVQuantilesShuffle( col_cond , lag_search , lag_keep )
		self.bc_method = bc_method
		self.bckwargs = bckwargs
		self._bcm = None
	##}}}
	
	def fit( self , Y0 , X0 , X1 = None ):##{{{
		"""
		Fit the AR2D2 model
		
		Parameters
		----------
		Y0 : np.array[ shape = (n_samples,n_features) ]
			Reference dataset during period 0
		X0 : np.array[ shape = (n_samples,n_features) ]
			Biased dataset during period 0
		X1	: np.array[ shape = (n_samples,n_features) ] or None
			Biased dataset during period 1. If None, the method is considered as
			stationary
		"""
		self.mvq.fit(Y0)
		self._bcm = self.bc_method(**self.bckwargs)
		if X1 is None:
			self._bcm.fit( Y0 , X0 )
		else:
			self._bcm.fit( Y0 , X0 , X1 )
	##}}}
	
	def predict( self , X1 = None , X0 = None ):##{{{
		"""
		Perform the bias correction
		Return Z1 if X0 is None (and vice-versa), else return a tuple Z1,Z0
		
		Parameters
		----------
		X1  : np.array[ shape = (n_samples,n_features) ]
			Array of value to be corrected in projection period
		X0  : np.array[ shape = (n_samples,n_features) ] or None
			Array of value to be corrected in calibration period
		
		Returns
		-------
		Z1 : np.array[ shape = (n_sample,n_features) ]
			Return an array of correction in projection period
		Z0 : np.array[ shape = (n_sample,n_features) ] or None
			Return an array of correction in calibration period
		"""
		if X0 is None and X1 is None:
			return
		if X0 is None:
			Z1b = self._bcm.predict(X1)
			return self.mvq.transform(Z1b)
		if X1 is None:
			Z0b = self._bcm.predict(X0)
			return self.mvq.transform(Z0b)
		Z1b,Z0b = self._bcm.predict(X1,X0)
		return self.mvq.transform(Z1b),self.mvq.transform(Z0b)
	##}}}
