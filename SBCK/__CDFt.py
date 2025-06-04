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

##################################################################################
##################################################################################
##                                                                              ##
## Original author  : Mathieu Vrac                                              ##
## Contact          : mathieu.vrac@lsce.ipsl.fr                                 ##
##                                                                              ##
## Notes   : CDFt is the re-implementation of the function CDFt of R package    ##
##           "CDFt" developped by Mathieu Vrac, available at                    ##
##           https://cran.r-project.org/web/packages/CDFt/index.html            ##
##           This code is governed by the GNU-GPL3 license with the             ##
##           authorization of Mathieu Vrac                                      ##
##                                                                              ##
##################################################################################
##################################################################################


###############
## Libraries ##
###############

import itertools as itt

import numpy       as np
import scipy.stats as sc
import scipy.interpolate as sci

from .__AbstractBC import UnivariateBC
from .__AbstractBC import MultiUBC

from .stats.__rv_extend import rv_base
from .stats.__rv_extend import rv_empirical

from typing import Self
from typing import Sequence
from .__AbstractBC import _rv_type
from .__AbstractBC import _mrv_type


###########
## Class ##
###########

class Univariate_CDFt(UnivariateBC):##{{{
	
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
	norm: str
	oob: str
	_oob_pmin: float
	_oob_pmax: float
	_oob_NCC: int
	
	def __init__( self , rvY0: _rv_type = rv_empirical , rvX0: _rv_type = rv_empirical , rvX1: _rv_type = rv_empirical , norm: str = "dynamical" , oob: str = "Y0" , **kwargs ) -> None:##{{{
		
		super().__init__( "Univariate_CDFt" , "NS" )
		
		self._typeY0,self._freezeY0,self.rvY0 = self._init(rvY0)
		self._typeX0,self._freezeX0,self.rvX0 = self._init(rvX0)
		self._typeX1,self._freezeX1,self.rvX1 = self._init(rvX1)
		
		self.oob       = oob
		self._oob_pmin = kwargs.get("oob_pmin",1e-2)
		self._oob_pmax = kwargs.get("oob_pmax",1-1e-2)
		self._oob_NCC  = kwargs.get("oob_NCC",5)
		if self.oob.startswith('CC') and len(self.oob) > 2 and self.oob[2:].isdigit():
			self._oob_NCC = int(self.oob[2:])
			self.oob      = 'CC'
		if self.oob.lower() not in ["none","y0","y0cc","cc"]:
			raise ValueError("Oob must be 'None', 'Y0', 'Y0CC' or 'CC'")
		
		self.norm   = norm
		if self.norm.lower() not in ["none","origin","dynamical"]:
			raise ValueError(f"Normalization must be 'None', 'origin' or 'dynamical' (= {self.norm})")

	##}}}
	
	def _normalization( self , Y0: np.ndarray , X0: np.ndarray , X1: np.ndarray ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:##{{{
		
		mY0 = Y0.mean()
		mX0 = X0.mean()
		mX1 = X1.mean()
		sY0 = Y0.std()
		sX0 = X0.std()
		sX1 = X1.std()
		NY0 = Y0
		
		match self.norm.lower():
			case 'none':
				NX0 = X0
				NX1 = X1
			case 'origin':
				NX0 = (X0 - mX0) * sY0 / sX0 + mY0
				NX1 = (X1 - mX1) * sY0 / sX0 + mY0 + (mX1 - mX0)
			case 'dynamical':
				NX0 = (X0 - mX0) * sY0 / sX0 + mY0
				NX1 = (X1 - mX1) * sY0 / sX0 + mY0 + (mX1 - mX0) * sY0 / sX0
		return NY0,NX0,NX1
	##}}}
	
	## Out of Bounds method ##{{{
	
	def _find_support( self , rvY0: rv_base , rvX0: rv_base , rvX1: rv_base ) -> tuple[np.ndarray,np.ndarray]:##{{{
		
		## First estimation of the support
		qmin = min([rv.icdf(0) for rv in [rvY0,rvX0,rvX1]])
		qmax = max([rv.icdf(1) for rv in [rvY0,rvX0,rvX1]])
		dq   = 0.05 * (qmax - qmin)
		nq   = 1000
		q    = np.linspace( qmin - dq , qmax + dq , nq )
		
		## Find the associated probabilities
		cdf = lambda q: rvY0.cdf( rvX0.icdf( rvX1.cdf( q ) ) )
		p   = cdf(q)
		
		## Cut the support
		i0 = max( np.sum(p == p[ 0]) - 1 , 0 )
		i1 = p.size - np.sum(p == p[-1])
		q  = np.linspace( q[i0] , q[i1] , nq )
		p  = cdf(q)
		
		return q,p
	##}}}
	
	def _oob_none( self , NY0: np.ndarray , NX0: np.ndarray , NX1: np.ndarray ) -> rv_base:##{{{
		
		rvNY0  = self._fit( NY0 , self._typeY0 , self._freezeY0 , self.rvY0 )
		rvNX0  = self._fit( NX0 , self._typeX0 , self._freezeX0 , self.rvX0 )
		rvNX1  = self._fit( NX1 , self._typeX1 , self._freezeX1 , self.rvX1 )
		c_icdf = lambda p: rvNX1.icdf( rvNX0.cdf(  rvNY0.icdf( p ) ) )
		
		p = np.linspace( 0 , 1 , 1000 )
		q = c_icdf(p)
		
		cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (p[0],p[-1]) )
		icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (q[0],q[-1]) )
		
		return rv_empirical( cdf , icdf , None )
	##}}}
	
	def _oob_Y0( self , NY0: np.ndarray , NX0: np.ndarray , NX1: np.ndarray ) -> rv_base:##{{{
		
		rvNY0  = self._fit( NY0 , self._typeY0 , self._freezeY0 , self.rvY0 )
		rvNX0  = self._fit( NX0 , self._typeX0 , self._freezeX0 , self.rvX0 )
		rvNX1  = self._fit( NX1 , self._typeX1 , self._freezeX1 , self.rvX1 )
		
		q,p = self._find_support( rvNY0 , rvNX0 , rvNX1 )
		
		## Correct the left tail
		if p[0] > self._oob_pmin:
			qmin = q[0] - (rvNY0.icdf(p[0]) - rvNY0.icdf(0))
			qL   = np.linspace( qmin , q[0] , 1000 )
			pL   = rvNY0.cdf( np.linspace( rvNY0.icdf(0) , rvNY0.icdf(p[0]) , 1000 ) )
			p    = np.hstack( (pL,p) )
			q    = np.hstack( (qL,q) )
		
		## Correct the right tail
		if p[-1] < self._oob_pmax:
			qmax = q[-1] + (rvNY0.icdf(1) - rvNY0.icdf(p[-1]))
			qR   = np.linspace( q[-1] , qmax , 1000 )
			pR   = rvNY0.cdf( np.linspace( rvNY0.icdf(p[-1]) , rvNY0.icdf(1) , 1000 ) )
			p    = np.hstack( (p,pR) )
			q    = np.hstack( (q,qR) )
		
		cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (p[0],p[-1]) )
		icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (q[0],q[-1]) )
		
		return rv_empirical( cdf , icdf , None )
	##}}}
	
	def _oob_Y0CC( self , NY0: np.ndarray , NX0: np.ndarray , NX1: np.ndarray ) -> rv_base:##{{{
		
		rvNY0  = self._fit( NY0 , self._typeY0 , self._freezeY0 , self.rvY0 )
		rvNX0  = self._fit( NX0 , self._typeX0 , self._freezeX0 , self.rvX0 )
		rvNX1  = self._fit( NX1 , self._typeX1 , self._freezeX1 , self.rvX1 )
		
		q,p = self._find_support( rvNY0 , rvNX0 , rvNX1 )
		
		## Correct the left tail
		if p[0] > self._oob_pmin:
			r    = (rvNX1.icdf(p[0]) - rvNX1.icdf(0)) / (rvNX0.icdf(p[0]) - rvNX0.icdf(0))
			if r == 0:
				r = 1
			qmin = q[0] - (rvNY0.icdf(p[0]) - rvNY0.icdf(0)) * r
			qL   = np.linspace( qmin , q[0] , 1000 )
			pL   = rvNY0.cdf( np.linspace( rvNY0.icdf(0) , rvNY0.icdf(p[0]) , 1000 ) )
			p    = np.hstack( (pL,p) )
			q    = np.hstack( (qL,q) )
		
		## Correct the right tail
		if p[-1] < self._oob_pmax:
			r    = (rvNX1.icdf(1) - rvNX1.icdf(p[-1])) / (rvNX0.icdf(1) - rvNX0.icdf(p[-1]))
			if r == 0:
				r = 1
			qmax = q[-1] + (rvNY0.icdf(1) - rvNY0.icdf(p[-1])) * r
			qR   = np.linspace( q[-1] , qmax , 1000 )
			pR   = rvNY0.cdf( np.linspace( rvNY0.icdf(p[-1]) , rvNY0.icdf(1) , 1000 ) )
			p    = np.hstack( (p,pR) )
			q    = np.hstack( (q,qR) )
		
		cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (p[0],p[-1]) )
		icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (q[0],q[-1]) )
		
		return rv_empirical( cdf , icdf , None )
	##}}}
	
	def _oob_CC( self , NY0: np.ndarray , NX0: np.ndarray , NX1: np.ndarray ) -> rv_base:##{{{
		
		rvNY0  = self._fit( NY0 , self._typeY0 , self._freezeY0 , self.rvY0 )
		rvNX0  = self._fit( NX0 , self._typeX0 , self._freezeX0 , self.rvX0 )
		rvNX1  = self._fit( NX1 , self._typeX1 , self._freezeX1 , self.rvX1 )
		NCC    = self._oob_NCC
		
		q,p = self._find_support( rvNY0 , rvNX0 , rvNX1 )
		
		cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (-np.inf,np.inf) )
		icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (-np.inf,np.inf) )
		
		Z1   = icdf( rvNX1.cdf(NX1) )
		idxF = np.argsort(Z1.squeeze())
		idxF = idxF[np.isfinite(Z1.squeeze()[idxF])]
		idxL = idxF[:NCC]
		idxR = idxF[-NCC:]
		
		## Left tail
		iL = ~np.isfinite(Z1) & (Z1 < 0)
		if np.any(iL):
			## Find D
			D = np.sum( Z1[idxL] - NX1[idxL] ) / NCC
			
			## Apply factor
			Z1[iL] = NX1[iL] + D
		
		## Right tail
		iR = ~np.isfinite(Z1) & (Z1 > 0)
		if np.any(iR):
			## Find D
			D = np.sum( Z1[idxR] - NX1[idxR] ) / NCC
			
			## Apply factor
			Z1[iR] = NX1[iR] + D
		
		return rv_empirical.fit(Z1)
	##}}}
	
	##}}}
	
	## Fit / predict functions ##{{{

	def fit( self , Y0: np.ndarray , X0: np.ndarray , X1: np.ndarray ) -> Self:##{{{
		
		## Fit
		self.rvY0 = self._fit( Y0 , self._typeY0 , self._freezeY0 , self.rvY0 )
		self.rvX0 = self._fit( X0 , self._typeX0 , self._freezeX0 , self.rvX0 )
		self.rvX1 = self._fit( X1 , self._typeX1 , self._freezeX1 , self.rvX1 )
		
		## Normalization step
		NY0,NX0,NX1 = self._normalization( Y0 , X0 , X1 )
		
		## Find rvY1 with Out of Bounds conditions
		match self.oob:
			case "Y0":
				self.rvY1 = self._oob_Y0( NY0 , NX0 , NX1 )
			case "Y0CC":
				self.rvY1 = self._oob_Y0CC( NY0 , NX0 , NX1 )
			case "CC":
				self.rvY1 = self._oob_CC( NY0 , NX0 , NX1 )
			case _:
				self.rvY1 = self._oob_none( NY0 , NX0 , NX1 )

		return self
	##}}}
	
	def _predictZ0( self , X0: np.ndarray | None , reinfer_X0: bool = False , **kwargs ) -> np.ndarray | None:##{{{
		if X0 is None:
			return None
		
		cdfX0 = self.rvX0.cdf
		if reinfer_X0:
			rvX0 = self._fit( X0 , self._typeX0 , self._freezeX0 , self.rvX0 )
			cdfX0 = rvX0.cdf
		Z0 = self.rvY0.icdf( cdfX0(X0) )
		
		return Z0
	##}}}
	
	def _predictZ1( self , X1: np.ndarray | None  , reinfer_X1: bool = False , **kwargs ) -> np.ndarray | None:##{{{
		if X1 is None:
			return None
		
		cdfX1 = self.rvX1.cdf
		if reinfer_X1:
			rvX1 = self._fit( X1 , self._typeX1 , self._freezeX1 , self.rvX1 )
			cdfX1 = rvX1.cdf
		Z1 = self.rvY1.icdf( cdfX1(X1) )
		
		return Z1
	##}}}
	
	##}}}
	
##}}}

class CDFt(MultiUBC):##{{{
	
	"""
	SBCK.CDFt
	=========
	Description
	-----------
	Quantile Mapping bias corrector, taking account of an evolution of the
	distribution, see [1].
	
	Normalization
	-------------
	Data can be normalized before applying the CDFt correction. Available
	methods are:
	- 'None' : No normalization,
	- 'mean' : Change the mean of X0 to that of Y0. The change in mean between
	  X0 and X1 is preserved.
	- 'meanstd': same as mean, but change also the standard deviation.
	- 'minmax': map the support of X0 to Y0. Apply the same transformation to X1
	- 'quant': Same as 'minmax', but instead of min and max, use some extreme
	  quantiles given by the parameter 'e'.
	- 'origin' original normalization use in old versions (< 2.0.0) of SBCK.
	
	Out Of Bounds
	-------------
	Correct the tails of the corrections. Available methods are:
	- 'None': no change,
	- 'CCN': Apply the delta change of the mean of the N last valids values to
	  the tail.
	- 'Y0': Copy the tail of the reference.
	- 'Y0CC': Copy a scaled tail of Y0, such that the change between the tail
	  of Y0 and Z1 is the change between X0 and X1.
	
	References
	----------
	[1] Michelangeli, P.-A., Vrac, M., and Loukos, H.: Probabilistic downscaling
	approaches: Application to wind cumulative distribution functions, Geophys.
	Res. Lett., 36, L11708, https://doi.org/10.1029/2009GL038401, 2009.
	
	Notes
	-----
	CDFt is the re-implementation of the function CDFt of R package "CDFt"
	developped by Mathieu Vrac, available at
	https://cran.r-project.org/web/packages/CDFt/index.htmm
	"""
	
	def __init__( self , rvY0: _mrv_type = rv_empirical , rvX0: _mrv_type = rv_empirical , rvX1: _mrv_type = rv_empirical , norm: str | Sequence[str] = "dynamical" , oob: str | Sequence[str] = "Y0" , **kwargs ):##{{{
		"""
		SBCK.CDFt.__init__
		==================
		
		Arguments
		---------
		rvY0: type | rv_base | Sequence[type | rv_base]
			Law of references
		rvX0: type | rv_base | Sequence[type | rv_base]
			Law of models in calibration period
		rvX1: type | rv_base | Sequence[type | rv_base]
			Law of models in projection period
		norm: str
			Normalisation method
		oob: str
			Out Of Bounds conditions
		
		Optional arguments
		------------------
		norm_e: float
			Quantile used in the 'quant' method
		oob_pmin: float
			Minimal value of 'valid' quantile in oob.
		oob_pmax: float
			Maximal value of 'valid' quantile in oob.
		
		"""
		## And init upper class
		args   = tuple()
		kwargs = { **{ 'rvY0' : rvY0 , 'rvX0' : rvX0 , 'rvX1' : rvX1 , 'norm' : norm , 'oob' : oob } , **kwargs }
		super().__init__( "CDFt" , Univariate_CDFt , args = args , kwargs = kwargs )
	##}}}
	
##}}}

