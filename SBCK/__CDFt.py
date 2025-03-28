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
import scipy.interpolate as sci

from .__AbstractBC import UnivariateBC
from .__AbstractBC import MultiUBC
from .tools.__rv_extend import WrapperStatisticalDistribution
from .tools.__rv_extend import rv_empirical


###########
## Class ##
###########

class Univariate_CDFt(UnivariateBC):##{{{
	
	class OoB:##{{{
		
		def __init__( self , method , **kwargs ):##{{{
			self.method = str(method)
			self.pmin   = float(kwargs.get( "oob_pmin" ,     1e-6 ))
			self.pmax   = float(kwargs.get( "oob_pmax" , 1 - 1e-6 ))
			self.NCC    =   int(kwargs.get( "oob_NCC"  ,        1 ))
			if self.method[:2] == "CC":
				if len(self.method) > 2:
					self.NCC    = int(self.method[2:])
					self.method = "CC"
			if self.method not in ["None","CC","Y0","Y0CC"]:
				raise ValueError( f"Unknow Out Of Bounds method: '{method}'" )
		##}}}
		
	##}}}
	
	class Norm:##{{{
		
		def __init__( self , method , **kwargs ):##{{{
			self.method    = str(method)
			self.dynamical = self.method[:2] == "d-"
			self.e         = kwargs.get("norm_e")
			
			if self.dynamical:
				self.method = self.method[2:]
			
			if self.method not in ["None","mean","meanstd","quant","minmax","origin"]:
				raise ValueError( f"Unknow normalization method: '{method}'" )
			
		##}}}
		
	##}}}
	
	def __init__( self , rvY = rv_empirical , rvX = rv_empirical , norm = "origin" , oob = "Y0" , **kwargs ):##{{{
		
		super().__init__( "Univariate_CDFt" , "NS" )
		
		self._rvY = rvY
		self._rvX = rvX
		self.rvY0 = None
		self.rvY1 = None
		self.rvX0 = None
		self.rvX1 = None
		
		self.oob    = self.OoB(   oob , **kwargs )
		self.norm   = self.Norm( norm , **kwargs )
		self._tools = {}
		
	##}}}
	
	## Normalization methods ##{{{
	
	def _norm_default( self ):##{{{
		self._tools["Y0s"] = self._tools["Y0"]
		self._tools["X0s"] = self._tools["X0"]
		self._tools["X1s"] = self._tools["X1"]
	##}}}
	
	def _norm_origin(self):##{{{
		
		Y0 = self._tools["Y0"]
		X0 = self._tools["X0"]
		X1 = self._tools["X1"]
		
		mY0 = Y0.mean()
		mX0 = X0.mean()
		mX1 = X1.mean()
		sY0 = Y0.std()
		sX0 = X0.std()
		sX1 = X1.std()
		
		X0s = (X0 - mX0) * sY0 / sX0 + mY0
		X1s = (X1 - mX1) * sY0 / sX0 + mX1 + mY0 - mX0
		
		self._tools["Y0s"] = Y0
		self._tools["X0s"] = X0s
		self._tools["X1s"] = X1s
	
	##}}}
	
	def _norm_mean( self ):##{{{
		
		Y0 = self._tools["Y0"]
		X0 = self._tools["X0"]
		X1 = self._tools["X1"]
		
		mY0 = Y0.mean()
		mX0 = X0.mean()
		mX1 = X1.mean()
		
		normX0Y0 = lambda x: ( x - mX0 ) + mY0
		normX1X0 = lambda x: ( x - mX1 ) + mX0
		normX0X1 = lambda x: ( x - mX0 ) + mX1
		
		X0s = normX0Y0(X0)
		if not self.norm.dynamical:
			X1s = normX0Y0(X1)
		else:
			X1s = normX0X1(normX0Y0(normX1X0(X1)))
		
		self._tools["Y0s"] = Y0
		self._tools["X0s"] = X0s
		self._tools["X1s"] = X1s
	##}}}
	
	def _norm_meanstd( self ):##{{{
		
		Y0 = self._tools["Y0"]
		X0 = self._tools["X0"]
		X1 = self._tools["X1"]
		
		mY0 = Y0.mean()
		mX0 = X0.mean()
		mX1 = X1.mean()
		sY0 = Y0.std()
		sX0 = X0.std()
		sX1 = X1.std()
		
		normX0Y0 = lambda x: ( x - mX0 ) / sX0 * sY0 + mY0
		normX1X0 = lambda x: ( x - mX1 ) / sX1 * sX0 + mX0
		normX0X1 = lambda x: ( x - mX0 ) / sX0 * sX1 + mX1
		
		X0s = normX0Y0(X0)
		if not self.norm.dynamical:
			X1s = normX0Y0(X1)
		else:
			X1s = normX0X1(normX0Y0(normX1X0(X1)))
		
		self._tools["Y0s"] = Y0
		self._tools["X0s"] = X0s
		self._tools["X1s"] = X1s
	##}}}
	
	def _norm_quant( self ):##{{{
		
		Y0 = self._tools["Y0"]
		X0 = self._tools["X0"]
		X1 = self._tools["X1"]
		
		e = self.norm.e
		if not isinstance(e,float):
			e = 5 * max( 1 / Y0.size , 1 / X0.size , 1 / X1.size )
		
		lX0 = np.quantile( X0 , e )
		lY0 = np.quantile( Y0 , e )
		lX1 = np.quantile( X1 , e )
		
		uX0 = np.quantile( X0 , 1 - e )
		uY0 = np.quantile( Y0 , 1 - e )
		uX1 = np.quantile( X1 , 1 - e )
		
		normX0Y0 = lambda x: ( x - lX0 ) / ( uX0 - lX0 ) * ( uY0 - lY0 ) + lY0
		normX1X0 = lambda x: ( x - lX1 ) / ( uX1 - lX1 ) * ( uX0 - lX0 ) + lX0
		normX0X1 = lambda x: ( x - lX0 ) / ( uX0 - lX0 ) * ( uX1 - lX1 ) + lX1
		
		X0s = normX0Y0(X0)
		if not self.norm.dynamical:
			X1s = normX0Y0(X1)
		else:
			X1s = normX0X1(normX0Y0(normX1X0(X1)))
		
		self._tools["Y0s"] = Y0
		self._tools["X0s"] = X0s
		self._tools["X1s"] = X1s
	##}}}
	
	def _norm_minmax( self ):##{{{
		
		Y0 = self._tools["Y0"]
		X0 = self._tools["X0"]
		X1 = self._tools["X1"]
		
		lX0 = np.min(X0)
		lY0 = np.min(Y0)
		lX1 = np.min(X1)
		uX0 = np.max(X0)
		uY0 = np.max(Y0)
		uX1 = np.max(X1)
		
		normX0Y0 = lambda x: ( x - lX0 ) / ( uX0 - lX0 ) * ( uY0 - lY0 ) + lY0
		normX1X0 = lambda x: ( x - lX1 ) / ( uX1 - lX1 ) * ( uX0 - lX0 ) + lX0
		normX0X1 = lambda x: ( x - lX0 ) / ( uX0 - lX0 ) * ( uX1 - lX1 ) + lX1
		
		X0s = normX0Y0(X0)
		if not self.norm.dynamical:
			X1s = normX0Y0(X1)
		else:
			X1s = normX0X1(normX0Y0(normX1X0(X1)))
		
		self._tools["Y0s"] = Y0
		self._tools["X0s"] = X0s
		self._tools["X1s"] = X1s
	##}}}
	
	##}}}
	
	## Out of Bounds method ##{{{
	
	def _find_support(self):##{{{
		
		rvY0s = self._tools["rvY0s"]
		rvX0s = self._tools["rvX0s"]
		rvX1s = self._tools["rvX1s"]
		
		## First estimation of the support
		qmin = min([rv.icdf(0) for rv in [rvY0s,rvX0s,rvX1s]])
		qmax = max([rv.icdf(1) for rv in [rvY0s,rvX0s,rvX1s]])
		dq   = 0.05 * (qmax - qmin)
		nq   = 1000
		q    = np.linspace( qmin - dq , qmax + dq , nq )
		
		## Find the associated probabilities
		cdf = lambda q: rvY0s.cdf( rvX0s.icdf( rvX1s.cdf( q ) ) )
		p   = cdf(q)
		
		## Cut the support
		i0 = max( np.sum(p == p[ 0]) - 1 , 0 )
		i1 = p.size - np.sum(p == p[-1])
		q  = np.linspace( q[i0] , q[i1] , nq )
		p  = cdf(q)
		
		return q,p
	##}}}
	
	def _oob_default(self):##{{{
		rvY0s = self._tools["rvY0s"]
		rvX0s = self._tools["rvX0s"]
		rvX1s = self._tools["rvX1s"]
		cdf  = lambda q: rvY0s.cdf(  rvX0s.icdf( rvX1s.cdf(  q ) ) )
		icdf = lambda p: rvX1s.icdf( rvX0s.cdf(  rvY0s.icdf( p ) ) )
		self.rvY1 = WrapperStatisticalDistribution( rv_empirical( cdf = cdf , icdf = icdf , no_pdf = True ) )
	##}}}
	
	def _oob_Y0(self):##{{{
		
		rvY0s = self._tools["rvY0s"]
		rvX0s = self._tools["rvX0s"]
		rvX1s = self._tools["rvX1s"]
		
		q,p = self._find_support()
		
		## Correct the left tail
		if p[0] > self.oob.pmin:
			qmin = q[0] - (rvY0s.icdf(p[0]) - rvY0s.icdf(0))
			qL   = np.linspace( qmin , q[0] , 1000 )
			pL   = rvY0s.cdf( np.linspace( rvY0s.icdf(0) , rvY0s.icdf(p[0]) , 1000 ) )
			p    = np.hstack( (pL,p) )
			q    = np.hstack( (qL,q) )
		
		## Correct the right tail
		if p[-1] < self.oob.pmax:
			qmax = q[-1] + (rvY0s.icdf(1) - rvY0s.icdf(p[-1]))
			qR   = np.linspace( q[-1] , qmax , 1000 )
			pR   = rvY0s.cdf( np.linspace( rvY0s.icdf(p[-1]) , rvY0s.icdf(1) , 1000 ) )
			p    = np.hstack( (p,pR) )
			q    = np.hstack( (q,qR) )
		
		cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (p[0],p[-1]) )
		icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (q[0],q[-1]) )
		self.rvY1 = WrapperStatisticalDistribution( rv_empirical( cdf = cdf , icdf = icdf , no_pdf = True ) )
	##}}}
	
	def _oob_Y0CC(self):##{{{
		
		rvY0s = self._tools["rvY0s"]
		rvX0s = self._tools["rvX0s"]
		rvX1s = self._tools["rvX1s"]
		
		q,p = self._find_support()
		
		## Correct the left tail
		if p[0] > self.oob.pmin:
			r    = (rvX1s.icdf(p[0]) - rvX1s.icdf(0)) / (rvX0s.icdf(p[0]) - rvX0s.icdf(0))
			if r == 0: r = 1
			qmin = q[0] - (rvY0s.icdf(p[0]) - rvY0s.icdf(0)) * r
			qL   = np.linspace( qmin , q[0] , 1000 )
			pL   = rvY0s.cdf( np.linspace( rvY0s.icdf(0) , rvY0s.icdf(p[0]) , 1000 ) )
			p    = np.hstack( (pL,p) )
			q    = np.hstack( (qL,q) )
		
		## Correct the right tail
		if p[-1] < self.oob.pmax:
			r    = (rvX1s.icdf(1) - rvX1s.icdf(p[-1])) / (rvX0s.icdf(1) - rvX0s.icdf(p[-1]))
			if r == 0: r = 1
			qmax = q[-1] + (rvY0s.icdf(1) - rvY0s.icdf(p[-1])) * r
			qR   = np.linspace( q[-1] , qmax , 1000 )
			pR   = rvY0s.cdf( np.linspace( rvY0s.icdf(p[-1]) , rvY0s.icdf(1) , 1000 ) )
			p    = np.hstack( (p,pR) )
			q    = np.hstack( (q,qR) )
		
		cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (p[0],p[-1]) )
		icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (q[0],q[-1]) )
		self.rvY1 = WrapperStatisticalDistribution( rv_empirical( cdf = cdf , icdf = icdf , no_pdf = True ) )
	##}}}
	
	def _oob_CC(self):##{{{
		
		Y0s   = self._tools["Y0s"]
		X0s   = self._tools["X0s"]
		X1s   = self._tools["X1s"]
		rvY0s = self._tools["rvY0s"]
		rvX0s = self._tools["rvX0s"]
		rvX1s = self._tools["rvX1s"]
		NCC   = self.oob.NCC
		
		q,p = self._find_support()
		
		cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (-np.inf,np.inf) )
		icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (-np.inf,np.inf) )
		
		Z1   = icdf( rvX1s.cdf(X1s) )
		idxF = np.argsort(Z1.squeeze())
		idxF = idxF[np.isfinite(Z1.squeeze()[idxF])]
		idxL = idxF[:NCC]
		idxR = idxF[-NCC:]
		
		## Left tail
		iL = ~np.isfinite(Z1) & (Z1 < 0)
		if np.any(iL):
			## Find D
			D = np.sum( Z1[idxL] - X1s[idxL] ) / NCC
			
			## Apply factor
			Z1[iL] = X1s[iL] + D
		
		## Right tail
		iR = ~np.isfinite(Z1) & (Z1 > 0)
		if np.any(iR):
			## Find D
			D = np.sum( Z1[idxR] - X1s[idxR] ) / NCC
			
			## Apply factor
			Z1[iR] = X1s[iR] + D
		
		self.rvY1 = WrapperStatisticalDistribution().fit(Z1)
	##}}}
	
	##}}}
	
	## Fit / predict functions ##{{{
	
	def fit( self , Y0 , X0 , X1 ):##{{{
		
		## Init shared variable
		self._tools["Y0"]   = Y0
		self._tools["X0"]   = X0
		self._tools["X1"]   = X1
		self._tools["Y0s"]  = Y0
		self._tools["X0s"]  = X0
		self._tools["X1s"]  = X1
		self._tools["rvY0"]  = WrapperStatisticalDistribution(self._rvY)
		self._tools["rvX0"]  = WrapperStatisticalDistribution(self._rvX)
		self._tools["rvX1"]  = WrapperStatisticalDistribution(self._rvX)
		self._tools["rvY0s"] = WrapperStatisticalDistribution(self._rvY)
		self._tools["rvX0s"] = WrapperStatisticalDistribution(self._rvX)
		self._tools["rvX1s"] = WrapperStatisticalDistribution(self._rvX)
		
		## Normalization step
		match self.norm.method:
			case "mean":
				self._norm_mean()
			case "meanstd":
				self._norm_meanstd()
			case "quant":
				self._norm_quant()
			case "minmax":
				self._norm_minmax()
			case "origin":
				self._norm_origin()
			case _:
				self._norm_default()
		
		## Define CDF
		for K,s in itt.product(["Y0","X0","X1"],["","s"]):
			YX   = self._tools[f"{K}{s}"]
			rvYX = self._tools[f"rv{K}{s}"]
			rvYX.fit(YX)
			self._tools[f"rv{K}{s}"]  = rvYX
		self.rvY0  = self._tools["rvY0"]
		self.rvX0  = self._tools["rvX0"]
		self.rvX1  = self._tools["rvX1"]
		
		## Find rvY1 with Out of Bounds conditions
		match self.oob.method:
			case "Y0":
				self._oob_Y0()
			case "Y0CC":
				self._oob_Y0CC()
			case "CC":
				self._oob_CC()
			case _:
				self._oob_default()
		
		##
		del self._tools
		
		return self
	##}}}
	
	def _predictZ0( self , X0 , reinfer_X0 = False , **kwargs ):##{{{
		if X0 is None:
			return None
		
		cdfX0 = self.rvX0.cdf
		if reinfer_X0:
			rvX0 = WrapperStatisticalDistribution(self._rvX)
			rvX0.fit(X0)
			cdfX0 = rvX0.cdf
		Z0 = self.rvY0.icdf( cdfX0(X0) )
		
		return Z0
	##}}}
	
	def _predictZ1( self , X1 , reinfer_X1 = False , **kwargs ):##{{{
		if X1 is None:
			return None
		
		cdfX1 = self.rvX1.cdf
		if reinfer_X1:
			rvX1 = WrapperStatisticalDistribution(self._rvX)
			rvX1.fit(X1)
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
	
	def __init__( self , rvY = rv_empirical , rvX = rv_empirical , norm = "origin" , oob = "Y0" , **kwargs ):##{{{
		"""
		SBCK.CDFt.__init__
		==================
		
		Arguments
		---------
		rvY: SBCK.tools.<law> | scipy.stats.<law>
			Law of references
		rvX: SBCK.tools.<law> | scipy.stats.<law>
			Law of models
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
		## Build args for MultiUBC
		if not isinstance( rvY , (list,tuple) ):
			if isinstance( rvX , (list,tuple) ):
				rvY = [rvY for _ in range(len(rvX))]
			else:
				rvY = [rvY]
		if not isinstance( rvX , (list,tuple) ):
			if isinstance( rvY , (list,tuple) ):
				rvX = [rvX for _ in range(len(rvY))]
			else:
				rvX = [rvX]
		if not len(rvX) == len(rvY):
			raise ValueError( "Incoherent arguments between rvY and rvX" )
		args = [ (rvy,rvx) for rvy,rvx in zip(rvY,rvX) ]
		
		## Build kwargs for MultiUBC
		ncorr   = len(args)
		ikwargs = kwargs
		ikwargs["norm"] = norm
		ikwargs["oob"]  = oob
		kwargs = [{} for _ in range(ncorr)]
		for key in ikwargs:
			kwarg = ikwargs[key]
			if isinstance( kwarg , (list,tuple) ):
				if len(kwarg) == ncorr:
					for i in range(ncorr):
						kwargs[i][key] = kwarg[i]
				else:
					raise ValueError( f"Invalid format for kwargs '{key}'" )
			else:
				for i in range(ncorr):
					kwargs[i][key] = kwarg
		
		## And init upper class
		super().__init__( "CDFt" , Univariate_CDFt , args = args , kwargs = kwargs )
	##}}}
	
##}}}

