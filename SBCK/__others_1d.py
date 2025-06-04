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
from .__AbstractBC import UnivariateBC
from .__AbstractBC import MultiUBC
from .__QM import QM

from .stats.__rv_extend import rv_base
from .stats.__rv_extend import rv_empirical

from typing import Self
from typing import Sequence
from typing import Callable
from .__AbstractBC import _rv_type
from .__AbstractBC import _mrv_type


###########
## Class ##
###########

class Univariate_QDM(UnivariateBC):##{{{
	
	_rvY0: _rv_type
	_rvX0: _rv_type
	_rvX1: _rv_type
	_planX0Y0: QM | None
	_planX1Y1: QM | None
	_delta_method: Callable
	_idelta_method: Callable

	def __init__( self , rvY0: _rv_type = rv_empirical , rvX0: _rv_type = rv_empirical , rvX1: _rv_type = rv_empirical , delta: str = "additive" ) -> None:##{{{
		
		super().__init__( "dOTC1d" , "NS" )
		
		self._rvY0 = rvY0
		self._rvX0 = rvX0
		self._rvX1 = rvX1
		
		match delta:
			case "additive":
				self._delta_method  = np.add
				self._idelta_method = np.subtract
			case "multiplicative":
				self._delta_method  = np.multiply
				self._idelta_method = np.divide
			case _:
				raise ValueError("delta method must be 'additive' or 'multiplicative'")

		self._planX0Y0 = None
		self._planX1Y1 = None
		
	##}}}
	
	def fit( self , Y0: np.ndarray , X0: np.ndarray , X1: np.ndarray ) -> Self:##{{{
		
		## Inference of Y1
		D0  = QM( rvY0 = self._rvX0 , rvX0 = self._rvY0 ).fit( X0 , Y0 ).predict(Y0)
		D1  = QM( rvY0 = self._rvX1 , rvX0 = self._rvX0 ).fit( X1 , X0 ).predict(D0)
		D10 = self._delta_method(D1 , D0)
		Y1  = self._idelta_method( Y0 , D10 )
		
		##
		self._planX0Y0 = QM( rvY0 = self._rvY0 , rvX0 = self._rvX0 ).fit( Y0 , X0 )
		self._planX1Y1 = QM(                     rvX0 = self._rvX1 ).fit( Y1 , X1 )
		
		return self
	##}}}
	
	def _predictZ1( self , X1: np.ndarray | None , reinfer_X1: bool = False , **kwargs ) -> np.ndarray | None:##{{{
		if X1 is None:
			return None
		if reinfer_X1:
			Z1 = QM( rvY0 = self._planX1Y1._rvY0 ).fit(None,X1).predict(X1)
		else:
			Z1 = self._planX1Y1.predict(X1)
		return Z1
	##}}}
	
	def _predictZ0( self , X0: np.ndarray | None , reinfer_X0: bool = False , **kwargs ) -> np.ndarray | None:##{{{
		if X0 is None:
			return None
		if reinfer_X0:
			Z0 = QM( rvY0 = self._planX0Y0._rvY0 ).fit(None,X0).predict(X0)
		else:
			Z0 = self._planX0Y0.predict(X0)
		return Z0
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
	
	def __init__( self , rvY0: _mrv_type = rv_empirical , rvX0: _mrv_type = rv_empirical , rvX1: _mrv_type = rv_empirical , delta: str = "additive" ):##{{{
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
		
		## And init upper class
		args   = tuple()
		kwargs = { 'delta' : delta , 'rvY0' : rvY0 , 'rvX0' : rvX0 }
		super().__init__( "QDM" , Univariate_QDM , args = args , kwargs = kwargs )
	##}}}
	
##}}}

class Univariate_QQD(UnivariateBC):##{{{
	
	_typeY0: type
	_typeX0: type
	_typeX1: type
	_freezeY0: bool
	_freezeX0: bool
	_freezeX1: bool
	rvY0: rv_base | None
	rvX0: rv_base | None
	rvX1: rv_base | None
	rvY1: rv_base | None
	
	def __init__( self , rvY0: _rv_type = rv_empirical , rvX0: _rv_type = rv_empirical , rvX1: _rv_type = rv_empirical , p_left: float = 0.01 , p_right: float = 0.99 ) -> None:##{{{
		
		super().__init__( "Univariate_QQ" , "NS" )
		
		self._typeY0,self._freezeY0,self.rvY0 = self._init(rvY0)
		self._typeX0,self._freezeX0,self.rvX0 = self._init(rvX0)
		self._typeX1,self._freezeX1,self.rvX1 = self._init(rvX1)
		
		self.p_left  = p_left
		self.p_right = p_right
		self._corr_left  = 0
		self._corr_right = 0
		
	##}}}
	
	def fit( self , Y0: np.ndarray , X0: np.ndarray , X1: np.ndarray ) -> Self:##{{{
		
		self.rvY0 = self._fit( Y0 , self._typeY0 , self._freezeY0 , self.rvY0 )
		self.rvX0 = self._fit( X0 , self._typeX0 , self._freezeX0 , self.rvX0 )
		self.rvX1 = self._fit( X1 , self._typeX1 , self._freezeX1 , self.rvX1 )
		
		self._corr_left  = self.rvY0.icdf(self.p_left)  - self.rvX0.icdf(self.p_left)
		self._corr_right = self.rvY0.icdf(self.p_right) - self.rvX0.icdf(self.p_right)
		
		## First estimation of Y1
		cdfX1 = self.rvX0.cdf(X1)
		Y1    = self.rvY0.icdf(cdfX1)
		
		## Correction of left tail
		idxL = cdfX1 < self.p_left
		if idxL.any():
			Y1[idxL] = self.rvX0.icdf(self.rvX1.cdf(X1[idxL])) + self._corr_left
		
		## Correction of right tail
		idxR = cdfX1 > self.p_right
		if idxR.any():
			Y1[idxR] = self.rvX0.icdf(self.rvX1.cdf(X1[idxR])) + self._corr_right
		
		## And store cdf
		self.rvY1 = rv_empirical.fit(Y1)
		
		return self
	##}}}
	
	def _predictZ0( self , X0: np.ndarray | None , reinfer_X0: bool = False , **kwargs ) -> np.ndarray | None:##{{{
		
		if X0 is None:
			return None
		
		rvX0 = self.rvX0
		if reinfer_X0:
			rvX0 = self._fit( X0 , self._typeX0 , self._freezeX0 , self.rvX0 )
		Z0 = self.rvY0.icdf(rvX0.cdf(X0))
		
		return Z0
	##}}}
	
	def _predictZ1( self , X1: np.ndarray | None , reinfer_X1: bool = False  , **kwargs ) -> np.ndarray | None:##{{{
		
		if X1 is None:
			return None
		
		rvX1 = self.rvX1 ## Because in QQD rvX1 dont exist!!!
		if reinfer_X1:
			rvX1 = self._fit( X1 , self._typeX1 , self._freezeX1 , self.rvX1 )
		Z1 = self.rvY1.icdf(rvX1.cdf(X1))
		
		return Z1
	##}}}
	
##}}}

class QQD(MultiUBC):##{{{
	
	def __init__( self ,  rvY0: _mrv_type = rv_empirical , rvX0: _mrv_type = rv_empirical , rvX1: _mrv_type = rv_empirical , p_left: float = 0.01 , p_right: float = 0.99 ):##{{{
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
		
		## And init upper class
		args   = tuple()
		kwargs = { 'rvY0' : rvY0 , 'rvX0' : rvX0 , 'rvX1' : rvX1 , 'p_left' : p_left , 'p_right' : p_right }
		super().__init__( "QQD" , Univariate_QQD , args = args , kwargs = kwargs )
	##}}}
	
##}}}



