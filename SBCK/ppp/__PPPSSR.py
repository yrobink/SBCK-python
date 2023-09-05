
## Copyright(c) 2022, 2023 Yoann Robin
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
from .__PrePostProcessing import PrePostProcessing


###########
## Class ##
###########

class PPPSSR(PrePostProcessing): ##{{{
	"""
	SBCK.ppp.PPPSSR
	===============
	
	Apply the SSR transformation. The SSR transformation replace the 0 by a
	random values between 0 and the minimal non zero value (the threshold). The
	inverse transform replace all values lower than the threshold by 0. The
	threshold used for inverse transform is given by the keyword `isaved`, which
	takes the value `Y0` (reference in calibration period), or `X0` (biased in
	calibration period), or `X1` (biased in projection period)
	
	>>> ## Start with data
	>>> Y0,X0,X1 = SBCK.datasets.like_tas_pr(2000)
	>>> 
	>>> ## Define the PPP method
	>>> ppp = SBCK.ppp.PPPSSR( bc_method = SBCK.CDFt , cols = 2 )
	>>> 
	>>> ## And now the correction
	>>> ppp.fit(Y0,X0,X1)
	>>> Z1,Z0 = ppp.predict(X1,X0)
	"""
	
	def __init__( self , *args , cols = None , threshold = None , **kwargs ): ##{{{
		"""
		Constructor
		===========
		
		Arguments
		---------
		cols: [int or array of int]
			The columns to apply the SSR
			or "X1"
		threshold: None or float
			If a float, this is the threshold used instead of a threshold
			infered from data
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		PrePostProcessing.__init__( self , *args , **kwargs )
		self._Xn   = threshold
		self.Xn    = None
		self._cols = cols
		
		if cols is not None:
			self._cols = np.array( [cols] , dtype = int ).squeeze()
		
	##}}}
	
	def transform( self , X ):##{{{
		"""
    	Apply the SSR transform.
		"""
		
		if X.ndim == 1:
			X = X.reshape(-1,1)
		
		if self._cols is None:
			self._cols = np.array( [i for i in range(X.shape[1])] , dtype = int ).squeeze()
		cols = self._cols
		
		Xn = np.array( [np.nanmin( np.where( X[:,cols] > 0 , X[:,cols] , np.nan ) , axis = 0 )] )
		
		if np.any(np.isnan(Xn)):
			Xn[np.isnan(Xn)] = 1
		
		if self._Xn is not None:
			Xn[:] = self._Xn
		
		if self.Xn is not None:
			Xn = self.Xn
		
		ncols = cols.size
		Xt = X.copy()
		Xt[:,cols] = np.where( (X[:,cols] > Xn).reshape(-1,ncols) , X[:,cols].reshape(-1,ncols) , np.random.uniform( low = Xn / 100 , high = Xn , size = (X.shape[0],ncols) ) ).squeeze()
		
		if self._kind == "Y0":
			self.Xn = Xn
		
		return Xt
	##}}}
	
	def itransform( self , Xt ):##{{{
		"""
    	Apply the SSR inverse transform.
		"""
		
		X = Xt.copy()
		if X.ndim == 1:
			X = X.reshape(-1,1)
		cols = self._cols
		X[:,cols] = np.where( Xt[:,cols] > self.Xn , Xt[:,cols] , 0 )
		
		return X
		##}}}
	
##}}}


