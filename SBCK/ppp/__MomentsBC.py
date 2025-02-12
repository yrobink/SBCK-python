
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

import deprecated
import numpy as np
from .__PrePostProcessing import PrePostProcessing


###########
## Class ##
###########

class DCS(PrePostProcessing):##{{{
	"""
	SBCK.ppp.DCS
	============
	
	DCS: Dynamical Center Scale
	
	Method used to ensure that the mean and std difference is the same after
	the correction.
	
	"""
	def __init__( self , *args , max_moment = 2 , **kwargs ):##{{{
		PrePostProcessing.__init__( self , *args , **kwargs )
		self._name      = "DCS"
		self.max_moment = max_moment
		self._m         = { "Y0" : 0 , "X0" : 0 , "X1" : 0 }
		self._s         = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		if not max_moment in [0,1,2]:
			raise ValueError("max_moment must be 0, 1 or 2")
	##}}}
	
	def transform( self , X ):##{{{
		if X.ndim == 1: X = X.reshape(-1,1)
		m = X.mean(0)
		s = X.std(0)
		self._m[self._kind] = m
		self._s[self._kind] = s
		
		match self.max_moment:
			case 0:
				Xt = X
			case 1:
				Xt = X - m
			case 2:
				Xt = (X - m) / s
		
		return Xt
	##}}}
	
	def itransform( self , Xt ):##{{{
		if Xt.ndim == 1: Xt = Xt.reshape(-1,1)
		m  = Xt.mean(0)
		s  = Xt.std(0)
		
		match self.max_moment:
			case 0:
				X = Xt
			case 1:
				X = Xt - m
				if self._kind == "X0":
					X = X + self._m["Y0"]
				if self._kind == "X1":
					X = X + self._m["Y0"] + self._m["X1"] - self._m["X0"]
			case 2:
				X = (Xt - m) / s
				if self._kind == "X0":
					X = X * self._s["Y0"] + self._m["Y0"]
				if self._kind == "X1":
					X = X * self._s["Y0"] / self._s["X0"] * self._s["X1"] + self._m["Y0"] + self._s["Y0"] / self._s["X0"] * (self._m["X1"] - self._m["X0"])
		
		return X
	##}}}
	
##}}}

class MNormalAdjust(PrePostProcessing):##{{{
	"""
	SBCK.ppp.MNormalAdjust
	======================
	
	Multivariate Normal Adjustement. Data are centered scale, then the mean and
	covariance matrix are corrected at the end.
	
	"""
	
	def __init__( self , *args , **kwargs ):##{{{
		PrePostProcessing.__init__( self , *args , **kwargs )
		self._name      = "MNormalAdjust"
		
		self._m = { "Y0" : 0 , "X0" : 0 , "X1" : 0 }
		self._c = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._s = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		self._i = { "Y0" : 1 , "X0" : 1 , "X1" : 1 }
		
	##}}}
	
	def transform( self , X ):##{{{
		
		k = self._kind
		
		self._m[k] = X.mean( axis = 0 ).reshape(-1,1)
		self._c[k] = np.cov(X.T)
		self._s[k] = np.linalg.cholesky(self._c[k])
		self._i[k] = np.linalg.pinv(self._s[k])
		
		Xt = self._i[k] @ ( X.T - self._m[k] )
		
		return Xt.T
	##}}}
	
	def itransform( self , Xt ):##{{{
		k = self._kind
		
		match k:
			case "Y0":
				X = self._s[k] @ ( Xt.T + self._m[k] )
			case "X0":
				k0 = "Y0"
				X = self._s[k0] @ ( Xt.T + self._m[k0] )
			case "X1":
				cF  = np.diag( np.diag(self._s["Y0"]) / np.diag(self._i["X0"]) )
				D01 = cF @ (self._m['X1'] - self._m['X0'])
				X   = self._s['X1'] @ self._i['X0'] @ self._s['Y0'] @ Xt.T + D01 + self._m['Y0']
		
		return X.T
		##}}}
	
##}}}


