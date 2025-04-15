
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

class UNormalAdjust(PrePostProcessing):##{{{
	"""
	SBCK.ppp.UNormalAdjust
	======================
	
	Univariate Normal Adjustement. Data are centered scale, then the mean and
	standard deviation are corrected at the end.
	
	"""
	def __init__( self , *args , recs_Xt = False , add_Xt1_bias = False , **kwargs ):##{{{
		PrePostProcessing.__init__( self , *args , **kwargs )
		self._name    = "UNormalAdjust"
		self._recs_Xt = recs_Xt
		self._add_Xt1_bias = add_Xt1_bias
		
		self._m  = { "Y0" : 0 , "X0" : 0 , "X1" : 0 }
		self._C  = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._s  = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._S  = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._is = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._iS = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		
	##}}}
	
	def transform( self , X ):##{{{
		
		k = self._kind
		
		self._m[k]  = X.mean( axis = 0 ).reshape(-1,1)
		self._C[k]  = np.cov(X.T)
		self._s[k]  = X.std( axis = 0 )
		self._is[k] = 1. / self._s[k]
		
		self._s[k]  = np.diag(self._s[k] )
		self._is[k] = np.diag(self._is[k])
		
		Xt = self._is[k] @ ( X.T - self._m[k] )
		
		return Xt.T
	##}}}
	
	def itransform( self , Xt ):##{{{
		k = self._kind
		
		if self._recs_Xt:
			mXt  = Xt.mean( axis = 0 ).reshape(-1,1)
			sXt  = Xt.std(  axis = 0 )
			isXt = 1. / sXt
			sXt  = np.diag( sXt)
			isXt = np.diag(isXt)
			Xt = ( isXt @ (Xt.T - mXt) ).T
		
		S = np.identity(Xt.shape[1])
		M = np.zeros( (Xt.shape[1],1) )
		if self._add_Xt1_bias:
			pass
		
		match k:
			case "Y0":
				X = self._s[k] @ Xt.T + self._m[k]
			case "X0":
				k0 = "Y0"
				X = self._s[k0] @ Xt.T + self._m[k0]
			case "X1":
				D01 = self._s["Y0"] @ self._is["X0"] @ (self._m['X1'] - self._m['X0'])
				X   = S @ self._s['X1'] @ self._is['X0'] @ self._s['Y0'] @ Xt.T + D01 + self._m['Y0'] + M
		
		return X.T
		##}}}
	
	
##}}}

class MNormalAdjust(PrePostProcessing):##{{{
	"""
	SBCK.ppp.MNormalAdjust
	======================
	
	Multivariate Normal Adjustement. Data are centered scale, then the mean and
	covariance matrix are corrected at the end.
	
	"""
	
	def __init__( self , *args , recs_Xt = False , **kwargs ):##{{{
		PrePostProcessing.__init__( self , *args , **kwargs )
		self._name    = "MNormalAdjust"
		self._recs_Xt = recs_Xt
		
		self._m  = { "Y0" : 0 , "X0" : 0 , "X1" : 0 }
		self._C  = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._s  = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._S  = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._is = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._iS = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		
	##}}}
	
	def transform( self , X ):##{{{
		
		k = self._kind
		
		self._m[k]  = X.mean( axis = 0 ).reshape(-1,1)
		self._C[k]  = np.cov(X.T)
		self._s[k]  = X.std( axis = 0 )
		self._is[k] = 1. / self._s[k]
		self._S[k]  = sqrtm(self._C[k])
		self._iS[k] = np.linalg.pinv(self._S[k])
		
		self._s[k]  = np.diag(self._s[k] )
		self._is[k] = np.diag(self._is[k])
		
		Xt = self._iS[k] @ ( X.T - self._m[k] )
		
		return Xt.T
	##}}}
	
	def itransform( self , Xt ):##{{{
		k = self._kind
		
		if self._recs_Xt:
			mXt  = Xt.mean( axis = 0 ).reshape(-1,1)
			C  = np.cov(Xt.T)
			S  = sqrtm(C)
			iS = np.linalg.pinv(S)
			Xt = ( iS @ (Xt.T - mXt) ).T
		
		match k:
			case "Y0":
				X = self._S[k] @ Xt.T + self._m[k]
			case "X0":
				k0 = "Y0"
				X = self._S[k0] @ Xt.T + self._m[k0]
			case "X1":
				cF  = self._s["Y0"] @ self._is["X0"]
				D01 = cF @ (self._m['X1'] - self._m['X0'])
				X   = self._S['X1'] @ self._iS['X0'] @ self._S['Y0'] @ Xt.T + D01 + self._m['Y0']
		
		return X.T
		##}}}
	
##}}}


