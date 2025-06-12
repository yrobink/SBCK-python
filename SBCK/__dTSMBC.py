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

from .__AbstractBC import AbstractBC
from .__dOTC import OTC
from .__dOTC import dOTC
from .__decorators import io_fit

from .stats.__shift import Shift

###########
## Class ##
###########

class TSMBC(AbstractBC):##{{{
	"""
	SBCK.TSMBC
	==========
	
	Description
	-----------
	Time Shifted Multivariate Bias Correction.
	
	References
	----------
	[1] Robin, Y. and Vrac, M.: Is time a variable like the others in
	multivariate statistical downscaling and bias correction?, Earth Syst.
	Dynam. Discuss. [preprint], https://doi.org/10.5194/esd-2021-12, in review,
	2021.
	"""
	
	def __init__( self , lag , bc_method = OTC , method = "row" , ref = "middle" , **kwargs ):##{{{
		"""
		Initialisation of TSMBC.
		
		Parameters
		----------
		lag       : integer
			Time lag of the shift
		bc_method : An class of SBCK
			bias correction method used, default is SBCK.OTC
		method    : string
			inverse method for shift, see SBCK.tools.Shift
		ref       : integer
			Reference columns/rows for inverse, see SBCK.tools.Shift, default is 0.5 * (lag+1)
		**kwargs  : arguments of bc_method
		
		Attributes
		----------
		bc_method : An element of SBCK
			Bias correction method
		shift     : Shift class
			class used to shift and un-shift data
		"""
		super().__init__( "TSMBC" , "S" )
		self.bc_method = bc_method(**kwargs)
		if ref == "middle": ref = int(0.5*(lag+1))
		self.shift     = Shift( lag , method , ref )
	##}}}
	
	## Properties {{{
	
	@property
	def ref(self):
		return self.shift.ref
	
	@ref.setter
	def ref( self , _ref ):
		self.shift.ref = _ref
	
	@property
	def method(self):
		return self.shift.method
	
	@method.setter
	def method( self , _method ):
		self.shift.method = _method
	##}}}
	
	def fit( self , Y0 , X0 ):##{{{
		"""
		Fit of the bc_method model on shifted X0 and Y0
		
		Parameters
		----------
		Y0	: np.ndarray
			Reference dataset
		X0	: np.ndarray
			Biased dataset
		"""
		Xs = self.shift.transform(X0)
		Ys = self.shift.transform(Y0)
		self.bc_method.fit( Ys , Xs )
		
		return self
	##}}}
	
	def _predictZ0( self , X0 , **kwargs ):##{{{
		if X0 is None:
			return None
		Xs = self.shift.transform(X0)
		return self.shift.inverse( self.bc_method.predict( Xs , **kwargs ) )
	##}}}
	
##}}}

class dTSMBC(AbstractBC):##{{{
	"""
	SBCK.dTSMBC
	===========
	
	Description
	-----------
	Time Shifted Multivariate Bias Correction where observations are unknown.
	
	References
	----------
	[1] Robin, Y. and Vrac, M.: Is time a variable like the others in
	multivariate statistical downscaling and bias correction?, Earth Syst.
	Dynam. Discuss. [preprint], https://doi.org/10.5194/esd-2021-12, in review,
	2021.
	"""
	
	def __init__( self , lag , bc_method = dOTC , method = "row" , ref = "middle" , **kwargs ):##{{{
		"""
		Initialisation of dTSMBC.
		
		Parameters
		----------
		lag       : integer
			Time lag of the shift
		bc_method : An element of SBCK
			bias correction method used, default is SBCK.dOTC()
		method    : string
			inverse method for shift, see SBCK.tools.Shift
		ref       : integer
			Reference columns/rows for inverse, see SBCK.tools.Shift, default is 0.5 * (lag+1)
		**kwargs  : arguments of bc_method
		
		Attributes
		----------
		bc_method : An element of SBCK
			Bias correction method
		shift     : Shift class
			class used to shift and un-shift data
		"""
		super().__init__( "dTSMBC" , "NS" )
		self.bc_method = bc_method(**kwargs)
		if ref == "middle": ref = int(0.5*(lag+1))
		self.shift     = Shift( lag , method , ref )
	##}}}
	
	## Methods and properties ##{{{
	
	@property
	def ref(self):
		return self.shift.ref
	
	@ref.setter
	def ref( self , _ref ):
		self.shift.ref = _ref
	
	@property
	def method(self):
		return self.shift.method
	
	@method.setter
	def method( self , _method ):
		self.shift.method = _method
	##}}}
	
	@io_fit
	def fit( self , Y0 , X0 , X1 ):##{{{
		"""
		Fit of the bc_method model on shifted X1, with learning shifted pair of Y0 and X0
		
		Parameters
		----------
		Y0	: np.ndarray
			Reference dataset on learning part
		X0	: np.ndarray
			Biased dataset on learning part
		X1	: np.ndarray
			Biased dataset on projection part
		"""
		Y0s = self.shift.transform(Y0)
		X0s = self.shift.transform(X0)
		X1s = self.shift.transform(X1)
		self.bc_method.fit( Y0s , X0s , X1s )
		
		return self
	##}}}
	
	def _predictZ0( self , X0 , **kwargs ):##{{{
		if X0 is None:
			return None
		X0s = self.shift.transform(X0)
		Z0s = self.bc_method._predictZ0( X0s , **kwargs )
		Z0  = self.shift.inverse(Z0s)
		
		return Z0
	##}}}
	
	def _predictZ1( self , X1 , **kwargs ):##{{{
		if X1 is None:
			return None
		X1s = self.shift.transform(X1)
		Z1s = self.bc_method._predictZ1( X1s , **kwargs )
		Z1  = self.shift.inverse(Z1s)
		return Z1
	##}}}
	
##}}}



