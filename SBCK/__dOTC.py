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

import numpy       as np

from .__AbstractBC import AbstractBC
from .__AbstractBC import MultiUBC
from .__QM         import QM
from .__decorators import io_fit
from .__decorators import io_predict

from .tools.__SparseHist import SparseHist
from .tools.__stats      import bin_width_estimator
from .tools.__OT         import POTemd
from .tools.__linalg import sqrtm
from .tools.__linalg import choleskym


###########
## Class ##
###########

class OTC(AbstractBC):##{{{
	"""
	SBCK.OTC
	========
	
	Description
	-----------
	Optimal Transport bias Corrector, see [1]
	
	References
	----------
	[1] Robin, Y., Vrac, M., Naveau, P., Yiou, P.: Multivariate stochastic bias
	corrections with optimal transport, Hydrol. Earth Syst. Sci., 23, 773–786,
	2019, https://doi.org/10.5194/hess-23-773-2019
	"""
	
	def __init__( self , bin_width = None , bin_origin = None , ot = POTemd() ):##{{{
		"""
		Initialisation of Optimal Transport bias Corrector.
		
		Parameters
		----------
		bin_width  : np.array( [shape = (n_features) ] )
			Lenght of bins, see SBCK.SparseHist. If is None, it is estimated during the fit
		bin_origin : np.array( [shape = (n_features) ] )
			Corner of one bin, see SBCK.SparseHist. If is None, np.repeat(0,n_features) is used
		ot         : OT*Solver*
			A solver for Optimal transport, default is POTemd()
		
		Attributes
		----------
		muY	: SBCK.SparseHist
			Multivariate histogram of references
		muX	: SBCK.SparseHist
			Multivariate histogram of biased dataset
		"""
		
		super().__init__("OTC")
		self.muX = None
		self.muY = None
		self.bin_width  = bin_width
		self.bin_origin = bin_origin
		self._plan       = None
		self._ot         = ot
	##}}}
	
	@io_fit
	def fit( self , Y0 , X0 ):##{{{
		"""
		Fit the OTC model
		
		Parameters
		----------
		Y0	: np.ndarray
			Reference dataset
		X0	: np.ndarray
			Biased dataset
		"""
		
		## Sparse Histogram
		self.bin_width  = np.array( [self.bin_width ] ).ravel() if self.bin_width  is not None else bin_width_estimator( [Y0,X0] )
		self.bin_origin = np.array( [self.bin_origin] ).ravel() if self.bin_origin is not None else np.zeros( self.bin_width.size )
		
		self.bin_width  = np.array( [self.bin_width] ).ravel()
		self.bin_origin = np.array( [self.bin_origin] ).ravel()
		
		self.muY = SparseHist( Y0 , bin_width = self.bin_width , bin_origin = self.bin_origin )
		self.muX = SparseHist( X0 , bin_width = self.bin_width , bin_origin = self.bin_origin )
		
		
		## Optimal Transport
		self._ot.fit( self.muX , self.muY )
		
		## 
		self._plan = np.copy( self._ot.plan() )
		self._plan = ( self._plan.T / self._plan.sum( axis = 1 ) ).T
		
		return self
	##}}}
	
	@io_predict
	def predict( self , X0 ):##{{{
		"""
		Perform the bias correction.
		
		Note: Only the center of the bins associated to the corrected points are
		returned, but all corrections of the form:
		>> otc.predict(X0) + np.random.uniform( low = - otc.bin_width / 2 , high = otc.bin_width / 2 , size = X0.shape[0] )
		are equivalent for OTC.
		
		Parameters
		----------
		X0  : np.ndarray
			Array of values to be corrected
		
		Returns
		-------
		Z0 : np.ndarray
			Return an array of correction
		"""
		indx = self.muX.argwhere(X0)
		indy = np.zeros_like(indx)
		for i,ix in enumerate(indx):
			indy[i] = np.random.choice( range(self.muY.sizep) , p = self._plan[ix,:] )
		return self.muY.c[indy,:]
	##}}}
	
##}}}

class dOTC(AbstractBC):##{{{
	"""
	SBCK.dOTC
	=========
	
	Description
	-----------
	Dynamical Optimal Transport bias Corrector, taking account of an evolution of the distribution. see [1]
	
	References
	----------
	[1] Robin, Y., Vrac, M., Naveau, P., Yiou, P.: Multivariate stochastic bias
	corrections with optimal transport, Hydrol. Earth Syst. Sci., 23, 773–786,
	2019, https://doi.org/10.5194/hess-23-773-2019
	"""
	
	def __init__( self , bin_width = None , bin_origin = None , cov_factor = "std" , ot = POTemd() ):##{{{
		"""
		Initialisation of dynamical Optimal Transport bias Corrector.
		
		Parameters
		----------
		bin_width  : np.array[ shape = (n_features) ] or None
			Lenght of bins, see SBCK.tools.SparseHist. If None, bin_width is estimated during fit.
		bin_origin : np.array[ shape = (n_features) ] or None
			Corner of one bin, see SBCK.tools.SparseHist. If None, np.repeat( 0 , n_features ) is used.
		cov_factor : str or np.array[ shape = (n_features,n_features) ]
			Correction factor during transfer of the evolution between X0 and X1 to Y0
				"cholesky" => compute the cholesky factor
				"sqrtm"    => compute the square root matrix factor
				"std"      => compute the standard deviation factor
				"id"       => identity is used
		ot         : OT*Solver*
			A solver for Optimal transport, default is SBCK.tools.OTNetworkSimplex()
		
		Attributes
		----------
		otc   : SBCK.OTC
			OTC corrector between X1 and the estimation of Y1
		"""
		super().__init__("dOTC")
		self._cov_factor_str = None
		self._cov_factor     = None
		if type(cov_factor) is str:
			if cov_factor not in ["cholesky","sqrtm","std","id"]:
				raise ValueError(f"'cov_factor' must be 'cholesky', 'sqrtm', 'std' or 'id'")
			self._cov_factor_str = cov_factor
		else:
			try:
				self._cov_factor = np.array([cov_factor])
			except:
				raise ValueError(f"cov_factor not a string and not castable to numpy array")
		
		self.bin_width  = bin_width
		self.bin_origin = bin_origin
		self._otcX0Y0   = None
		self.otc        = None
		self._ot        = ot
	##}}}
	
	@io_fit
	def fit( self , Y0 , X0 , X1 ):##{{{
		"""
		Fit the dOTC model to perform non-stationary bias correction during period 1. For period 0, see OTC
		
		Parameters
		----------
		Y0 : np.ndarray
			Reference dataset during calibration period
		X0 : np.ndarray
			Biased dataset during calibration period
		X1	: np.ndarray
			Biased dataset during projection period
		"""
		## Set the covariance factor correction
		
		if Y0.ndim == 1: Y0 = Y0.reshape(-1,1)
		if X0.ndim == 1: X0 = X0.reshape(-1,1)
		if X1.ndim == 1: X1 = X1.reshape(-1,1)
		
		if self._cov_factor is None:
			if self._cov_factor_str in ["std","sqrtm","cholesky"]:
				if Y0.shape[1] == 1:
					self._cov_factor = np.std( Y0 ) / np.std( X0 )
					self._cov_factor = np.array([self._cov_factor]).reshape(1,1)
				elif self._cov_factor_str == "cholesky":
					fact0 = choleskym( np.cov( Y0 , rowvar = False ) )
					fact1 = choleskym( np.cov( X0 , rowvar = False ) )
					self._cov_factor = np.dot( fact0 , np.linalg.inv( fact1 ) )
				elif self._cov_factor_str == "sqrtm":
					fact0 = sqrtm( np.cov( Y0 , rowvar = False ) )
					fact1 = sqrtm( np.cov( X0 , rowvar = False ) )
					self._cov_factor = np.dot( fact0 , np.linalg.inv( fact1 ) )
				else:
					fact0 = np.std( Y0 , axis = 0 )
					fact1 = np.std( X0 , axis = 0 )
					self._cov_factor = np.diag( fact0 / fact1 )
			else:
				self._cov_factor = np.identity(Y0.shape[1])
		self._cov_factor = self._cov_factor.reshape(self.ndim,self.ndim)
		self.bin_width = self.bin_width if self.bin_width is not None else bin_width_estimator( [Y0,X0,X1] )
		
		
		## Optimal plan
		otcY0X0 = OTC( self.bin_width , self.bin_origin , ot = self._ot )
		otcX0X1 = OTC( self.bin_width , self.bin_origin , ot = self._ot )
		otcY0X0.fit( X0 , Y0 )
		otcX0X1.fit( X1 , X0 )
		self._otcX0Y0 = OTC( self.bin_width , self.bin_origin , ot = self._ot )
		self._otcX0Y0.fit(Y0,X0)
		
		## Estimation of Y1
		yX0 = otcY0X0.predict(Y0)
		yX1 = otcX0X1.predict(yX0)
		motion = yX1 - yX0
		Y1 = Y0 + (self._cov_factor @ motion.T).T
		
		## Optimal plan for correction
		self.otc = OTC( self.bin_width , self.bin_origin , ot = self._ot )
		self.otc.fit( Y1 , X1 )
		
		return self
	##}}}
	
	@io_predict
	def predict( self , X1 , X0 = None ):##{{{
		"""
		Perform the bias correction
		Return Z1 if X0 is None, else return a tuple Z1,Z0
		
		Note: Only the center of the bins associated to the corrected points are
		returned, but all corrections of the form:
		>> dotc.predict(X1) + np.random.uniform( low = - dotc.bin_width / 2 , high = dotc.bin_width / 2 , size = X1.shape[0] )
		are equivalent for dOTC.
		
		Parameters
		----------
		X1  : np.ndarray
			Array of value to be corrected in projection period
		X0  : np.ndarray or None
			Array of value to be corrected in calibration period
		
		Returns
		-------
		Z1 : np.ndarray
			Return an array of correction in projection period
		Z0 : np.ndarray
			Return an array of correction in calibration period
		"""
		Z1 = self.otc.predict( X1 )
		if X0 is not None:
			Z0 = self._otcX0Y0.predict(X0)
			return Z1,Z0
		return Z1
	##}}}
	
##}}}


class Univariate_dOTC1d(AbstractBC):##{{{
	def __init__( self , cov_factor = "std" ):
		super().__init__( "dOTC1d" )
		self._planX0Y0 = None
		self._planX1Y1 = None
		self._cov_factor = cov_factor
	
	def fit( self , Y0 , X0 , X1 ):
		
		## cfactor
		cfactor = 1.
		if self._cov_factor == "std":
			cfactor  = Y0.std() / X0.std()
		
		## Inference of Y1
		planX0X1 = QM().fit( X1 , X0 )
		planY0X0 = QM().fit( X0 , Y0 )
		D0       = planY0X0.predict(Y0)
		D1       = planX0X1.predict(D0)
		dynamic  = cfactor * (D1 - D0)
		Y1       = Y0 + dynamic
		
		##
		self._planX0Y0 = QM().fit( Y0 , X0 )
		self._planX1Y1 = QM().fit( Y1 , X1 )
		
		return self
	
	def predict( self , X1 , X0 = None ):
		Z1 = self._planX1Y1.predict(X1)
		if X0 is None: return Z1
		Z0 = self._planX0Y0.predict(X0)
		return Z1,Z0
##}}}

class dOTC1d(MultiUBC):##{{{
	"""
	SBCK.dOTC1d
	===========
	
	Description
	-----------
	One dimensionnal version of dOTC, use quantile mapping (instead of simplex)
	to solve the transport problem (very very very faster). If cov_factor = 1 ,
	this is actually the QDM method.
	
	References
	----------
	[1] Robin, Y., Vrac, M., Naveau, P., Yiou, P.: Multivariate stochastic bias
	corrections with optimal transport, Hydrol. Earth Syst. Sci., 23, 773–786,
	2019, https://doi.org/10.5194/hess-23-773-2019
	"""
	def __init__( self , cov_factor = "std" ):
		super().__init__( "dOTC1d" , Univariate_dOTC1d , kwargs = { "cov_factor" : cov_factor } )
##}}}

