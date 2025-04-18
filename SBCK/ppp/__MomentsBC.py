
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

###############
## Libraries ##
###############

import numpy as np
from .__PrePostProcessing import PrePostProcessing

from ..tools.__linalg import sqrtm


###########
## Class ##
###########

class MNPar:##{{{
	
	def __init__( self , X , univariate = False ):##{{{
		if X.ndim == 1:
			X = X.reshape(-1,1)
		self.ndim = X.shape[1]
		self.univariate = univariate
		self._m   = X.mean(0).reshape(1,self.ndim)
		self._s   = X.std(0).reshape(1,self.ndim)
		self._C   = np.cov( X , rowvar = False ).reshape(self.ndim,self.ndim)
		self._S   = sqrtm(self.C)
		self._ivs = 1. / self.s
		self._ivS = np.linalg.pinv(self.S)
	##}}}
	
	## Normalization methods ##{{{
	
	def _unormalize( self , X ):
		return ( self.ivs * (X - self.m) )

	def _mnormalize( self , X ):
		return ( self.ivS @ (X - self.m).T ).T

	def normalize( self , X ):
		if self.univariate or self.ndim == 1:
			N = self._unormalize(X.reshape(-1,self.ndim))
		else:
			N = self._mnormalize(X.reshape(-1,self.ndim))
		return N.reshape(-1,self.ndim)
	
	def _iunormalize( self , X ):
		return self.s * X + self.m

	def _imnormalize( self , X ):
		return ( self.S @ X.T ).T + self.m

	def inormalize( self , N ):
		if self.univariate or self.ndim == 1:
			X = self._iunormalize(N.reshape(-1,self.ndim))
		else:
			X = self._imnormalize(N.reshape(-1,self.ndim))
		return X.reshape(-1,self.ndim)
	##}}}
	
	## Properties ##{{{
	
	@property
	def m(self):
		return self._m
	
	@property
	def s(self):
		return self._s
	
	@property
	def C(self):
		return self._C
	
	@property
	def S(self):
		return self._S
	
	@property
	def ivs(self):
		return self._ivs
	
	@property
	def ivS(self):
		return self._ivS
	
	##}}}
	
##}}}

class UMNAdjust(PrePostProcessing):##{{{
	
	def __init__( self , *args , univariate = False , **kwargs ):##{{{
		PrePostProcessing.__init__( self , *args , **kwargs )
		self._name      = "MNAdjust"
		self.univariate = univariate
		self._p         = {}
	##}}}
	
	def transform( self , X ):##{{{
		self._p[self._kind] = MNPar( X , univariate = self.univariate )
		NX  = self._p[self._kind].normalize(X)
		return NX
	##}}}
	
	def itransform( self , Xt ):##{{{
		
		pXt = MNPar( Xt , univariate = self.univariate )
		NXt = pXt.normalize(Xt)
		
		match self._kind:
			case 'X0':
				X = self._p['Y0'].inormalize(Xt)
			case 'X1':
				if self.univariate:
					X  =   self._p['X1'].s * self._p['X0'].ivs * self._p['Y0'].s * NXt
				else:
					X  = ( self._p['X1'].S @ self._p['X0'].ivS @ self._p['Y0'].S @ NXt.T ).T
				X  = X + self._p['Y0'].s * self._p['X0'].ivs * (self._p['X1'].m - self._p['X0'].m)
				X  = X + self._p['X1'].s * self._p['X0'].ivs * self._p['Y0'].s * pXt.ivs * pXt.m
				X  = X + self._p['Y0'].m
		
		return X
	##}}}
	
##}}}

