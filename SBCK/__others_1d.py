# -*- coding: utf-8 -*-

## Copyright(c) 2021 / 2024 Yoann Robin
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
from .__AbstractBC import UnivariateBC
from .__AbstractBC import MultiUBC
from .__QM import Univariate_QM
from .tools.__rv_extend import rv_empirical
from SBCK.tools.__rv_extend import WrapperStatisticalDistribution


###########
## Class ##
###########

class Univariate_QDM(UnivariateBC):##{{{
	
	def __init__( self , delta = "additive" , rvY = np.histogram , rvX = np.histogram ):##{{{
		super().__init__( "Univariate_QDM" , "NS" )
		self._delta_method  = np.add
		self._idelta_method = np.subtract
		if delta == "multiplicative":
			self._delta_method  = np.multiply
			self._idelta_method = np.divide
		if isinstance(delta,(list,tuple)):
			self._delta_method  = delta[0]
			self._idelta_method = delta[1]
		
		self._delta  = None
		self._qmX0Y0 = None
		self._qmX1Y0 = None
		self._qm_kwargs = { "rvY" : rvY , "rvX" : rvX }
	##}}}
	
	def fit( self , Y0 , X0 , X1 ):##{{{
		qmX0Y0 = Univariate_QM(**self._qm_kwargs).fit( Y0 , X0 )
		qmX1Y0 = Univariate_QM(**self._qm_kwargs).fit( Y0 , X1 )
		qmX1X0 = Univariate_QM(**self._qm_kwargs).fit( X0 , X1 )
		
		self._delta  = self._idelta_method( X1 , qmX1X0.predict(X1) ).reshape(-1,1)
		self._qmX0Y0 = qmX0Y0
		self._qmX1Y0 = qmX1Y0
		
		return self
	##}}}
	
	def _predictZ0( self , X0 , **kwargs ):##{{{
		if X0 is None:
			return None
		Z0 = self._qmX0Y0.predict(X0)
		return Z0
	##}}}
	
	def _predictZ1( self , X1 , **kwargs ):##{{{
		if X1 is None:
			return None
		Z1 = self._delta_method( self._qmX1Y0.predict(X1).reshape(-1,1) , self._delta.reshape(-1,1) ).reshape(X1.shape)
		return Z1
	##}}}
	
##}}}

class QDM(MultiUBC):##{{{
	
	"""
	SBCK.QDM
	========
	
	Description
	-----------
	QDM Bias correction method, see [1]
	
	References
	----------
	[1] Cannon, A. J., Sobie, S. R., and Murdock, T. Q.: Bias correction of
	simulated precipitation by quantile mapping: how well do methods preserve
	relative changes in quantiles and extremes?, J. Climate, 28, 6938–6959,
	https://doi.org/10.1175/JCLI-D-14- 00754.1, 2015.
	"""
	
	def __init__( self , delta = "additive" , rvY = rv_empirical , rvX = rv_empirical ):##{{{
		"""
		SBCK.QDM.__init__
		=================
		
		Arguments
		---------
		delta : str or tuple
			Delta method: "additive" or "multiplicative". It is possible to
			pass custom delta function with a tuple where the first element is
			the transform ( e.g. np.add or np.multiply) and the second one its
			inverse (e.g. np.subtract or np.divide)
		rvY: SBCK.tools.<law> | scipy.stats.<law>
			Law of references
		rvX: SBCK.tools.<law> | scipy.stats.<law>
			Law of models
		"""
		
		## Build args for MultiUBC
		if not isinstance( delta , (list,tuple) ):
			if isinstance( rvY , (list,tuple) ):
				delta = [delta for _ in range(len(rvY))]
			elif isinstance( rvX , (list,tuple) ):
				delta = [delta for _ in range(len(rvX))]
			else:
				delta = [delta]
		if not isinstance( rvY , (list,tuple) ):
			if isinstance( delta , (list,tuple) ):
				rvY = [rvY for _ in range(len(delta))]
			elif isinstance( rvX , (list,tuple) ):
				rvY = [rvY for _ in range(len(rvX))]
			else:
				rvY = [rvY]
		if not isinstance( rvX , (list,tuple) ):
			if isinstance( delta , (list,tuple) ):
				rvX = [rvX for _ in range(len(delta))]
			elif isinstance( rvY , (list,tuple) ):
				rvX = [rvX for _ in range(len(rvY))]
			else:
				rvX = [rvX]
		if not len(set({len(delta),len(rvX),len(rvY)})) == 1:
			raise ValueError( "Incoherent arguments between delta, rvY and rvX" )
		args = [ (dlta,rvy,rvx) for dlta,rvy,rvx in zip(delta,rvY,rvX) ]
		
		## And init upper class
		super().__init__( "QDM" , Univariate_QDM , args = args )
	##}}}
	
##}}}


class Univariate_QQD(UnivariateBC):##{{{
	
	def __init__( self , p_left = 0.01 , p_right = 0.99 ):##{{{
		super().__init__( "Univariate_QQ" , "NS" )
		self.p_left  = p_left
		self.p_right = p_right
		self._corr_left  = 0
		self._corr_right = 0
		
		self._rvX    = rv_empirical
		self._rvY    = rv_empirical
		self.rvY0 = WrapperStatisticalDistribution(self._rvY)
		self.rvX0 = WrapperStatisticalDistribution(self._rvX)
	##}}}
	
	def fit( self , Y0 , X0 , X1 ):##{{{
		
		self.rvX0.fit(X0)
		self.rvY0.fit(Y0)
		
		self._corr_left  = self.rvY0.icdf(self.p_left)  - self.rvX0.icdf(self.p_left)
		self._corr_right = self.rvY0.icdf(self.p_right) - self.rvX0.icdf(self.p_right)
		
		return self
	##}}}
	
	def _predictZ10( self , X , **kwargs ):##{{{
		if X is None:
			return None
		cdfX = self.rvX0.cdf(X)
		Z    = self.rvY0.icdf(cdfX)
		
		## Correction of left tail
		idxL = cdfX < self.p_left
		if idxL.any():
			Z[idxL] = X[idxL] + self._corr_left
		
		## Correction of right tail
		idxR = cdfX > self.p_right
		if idxR.any():
			Z[idxR] = X[idxR] + self._corr_right
		
		return Z
	##}}}
	
	def _predictZ0( self , X0 , **kwargs ):##{{{
		return self._predictZ10( X0 , **kwargs )
	##}}}
	
	def _predictZ1( self , X1 , **kwargs ):##{{{
		return self._predictZ10( X1 , **kwargs )
	##}}}
	
##}}}

class QQD(MultiUBC):##{{{
	
	def __init__( self , p_left = 0.01 , p_right = 0.99 ):##{{{
		"""
		SBCK.QQD.__init__
		=================
		
		QQD: Quantile-Quantile of Deque (2007). The method is a simple quantile
		mapping inferred in calibration and applied in projection. Quantiles
		below and above p_left and p_right are corrected by the constants
		cst_left and cst_right, defined by:
		
		cst_left  = rvY0.icdf(p_left)  - rvX0.icdf(p_left)
		cst_right = rvY0.icdf(p_right) - rvX0.icdf(p_right)
		
		Deque, 2007: doi:10.1016/j.gloplacha.2006.11.030
		
		Arguments
		---------
		p_left: float
			Minimal left quantile
		p_right: float
			Maximal right quantile
		"""
		
		## Build args for MultiUBC
		if not isinstance( p_left  , (list,tuple) ):
			if isinstance( p_right , (list,tuple) ):
				p_left = [p_left for _ in range(len(p_right))]
			else:
				p_left = [p_left]
		if not isinstance( p_right , (list,tuple) ):
			if isinstance( p_left , (list,tuple) ):
				p_right = [p_right for _ in range(len(p_left))]
			else:
				p_right = [p_right]
		if not len(p_left) == len(p_right):
			raise ValueError( f"Incoherent arguments between p_left and p_right" )
		args = [ (pL,pR) for pL,pR in zip(p_left,p_right) ]
		
		## And init upper class
		super().__init__( "QQD" , Univariate_QQD , args = args )
	##}}}
	
##}}}



