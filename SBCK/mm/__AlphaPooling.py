
## Copyright(c) 2024 Yoann Robin
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
import scipy.interpolate as sci
import scipy.optimize as sco

from ..__AbstractBC import AbstractBC
from ..__AbstractBC import MultiUBC
from ..__QM import Univariate_QM
from ..tools.__rv_extend import rv_empirical
from ..tools.__rv_extend import WrapperStatisticalDistribution


#############
## Classes ##
#############

class Univariate_AlphaPooling(AbstractBC):##{{{
	
	def __init__( self , alpha = None , rvY = rv_empirical , rvX = rv_empirical ):##{{{
		super().__init__( "Univariate_AlphaPooling" )
		self._rvY = rvY
		self._rvX = rvX
		
		self._rvY0 = None
		self._rvY1 = None
		self._qm0  = None
		self._qm1  = None
		
		self._fa     = False
		self._a      = None
		self._igpool = None
		if isinstance(alpha,(float,int)):
			self._fa = True
			self.a   = alpha
		self._w      = None
		
	##}}}
	
	def _normalization( self , lX1 , lX0 , Y0 ):##{{{
		
		lX1s = []
		lX0s = []
		for X1,X0 in zip(lX1,lX0):
			
			e = 5 * max( 1 / Y0.size , 1 / X0.size , 1 / X1.size )
			
			X0l = np.quantile( X0 , e )
			Y0l = np.quantile( Y0 , e )
			X1l = np.quantile( X1 , e )
			
			X0u = np.quantile( X0 , 1 - e )
			Y0u = np.quantile( Y0 , 1 - e )
			X1u = np.quantile( X1 , 1 - e )
			
			normX0Y0 = lambda x: ( x - X0l ) / ( X0u - X0l ) * ( Y0u - Y0l ) + Y0l
			normX1X0 = lambda x: ( x - X1l ) / ( X1u - X1l ) * ( X0u - X0l ) + X0l
			normX0X1 = lambda x: ( x - X0l ) / ( X0u - X0l ) * ( X1u - X1l ) + X1l
			
			X1s = normX0X1(normX0Y0(normX1X0(X1)))
			X0s = normX0Y0(X0)
			
			lX1s.append(X1s)
			lX0s.append(X0s)
		
		
		return lX1s,lX0s
	##}}}
	
	def _gpool( self , x ):##{{{
		return ( x**self.a - (1-x)**self.a ) / self.a
	##}}}
	
	def _pooling( self , lrvX ):##{{{
		
		## Find support
		xmin =  1e9
		xmax = -1e9
		for rvX in lrvX:
			xmin = min( xmin , rvX.icdf(1e-6) )
			xmax = max( xmax , rvX.icdf(1-1e-6) )
		x   = np.linspace( xmin , xmax , 1000 )
		
		## And find the combination
		zg   = np.array( [ self._w[i] * (rvX.cdf(x)**self._a - (1-rvX.cdf(x))**self._a) for i,rvX in enumerate(lrvX) ] ).sum(0) / self._a
		y    = self._igpool(zg)
		cdf  = sci.interp1d( x , y , bounds_error = False , fill_value = (0,1) )
		icdf = sci.interp1d( y , x , bounds_error = False , fill_value = (xmin,xmax) )
		rvG  = rv_empirical( cdf = cdf , icdf = icdf , no_pdf = True )
		
		return rvG
	##}}}
	
	def _fit_pooling_optim( self , p , rvY , lrvX ):##{{{
		
		if not self._fa:
			self.a = np.exp(p[0])
			self.w = np.exp(p[1:])
		else:
			self.w = np.exp(p)
		rvG    = self._pooling( lrvX )
		
		xmin = min( [rv.icdf(  1e-6) for rv in [rvY,rvG]] )
		xmax = max( [rv.icdf(1-1e-6) for rv in [rvY,rvG]] )
		x    = np.linspace( xmin , xmax , 1000 )
		
		return np.sum( np.abs(rvY.cdf(x) - rvG.cdf(x))**2 )
	##}}}
	
	def _fit_pooling( self , rvY , lrvX ):##{{{
		
		## Parameters
		nmod = len(lrvX)
		
		## Find starting point
		if self._fa:
			x0    = np.log(np.ones(nmod))
		else:
			x0    = np.log(np.ones(nmod+1))
			x0[0] = np.log(0.5)
			s0    = self._fit_pooling_optim( x0 , rvY , lrvX )
			x0[0] = np.log(2)
			s1    = self._fit_pooling_optim( x0 , rvY , lrvX )
			x0[0] = np.log( 0.5 if s0 < s1 else 2 )
		
		## Optimization
		success = False
		xn      = x0.copy()
		while not success:
			res     = sco.minimize( self._fit_pooling_optim , xn , args = (rvY,lrvX) )
			if np.sum( np.abs(x0-xn) ) < 1e-3:
				break
			x0      = xn.copy()
			xn      = res.x
			success = res.success
		
		## Set values
		if self._fa:
			self.w = np.exp(res.x)
		else:
			self.a = np.exp(res.x[0])
			self.w = np.exp(res.x[1:])
	##}}}
	
	def fit( self , Y0 , *args ):##{{{
		
		## Check number of arguments
		nargs = len(args)
		if not nargs % 2 == 0:
			raise ValueError( "Calibration and projection of each model must be given!")
		
		## Split in calibration / projection
		nX  = nargs // 2
		lX0 = args[:nX]
		lX1 = args[nX:]
		
		## Normalization
		lX1s,lX0s = self._normalization(lX1,lX0,Y0)
		
		## Build random variables
		rvY0   =   WrapperStatisticalDistribution(self._rvY).fit(Y0)
		lrvX0  = [ WrapperStatisticalDistribution(self._rvX).fit(X0)  for X0  in lX0 ]
		lrvX1  = [ WrapperStatisticalDistribution(self._rvX).fit(X1)  for X1  in lX1 ]
		lrvX0s = [ WrapperStatisticalDistribution(self._rvX).fit(X0s) for X0s in lX0s ]
		lrvX1s = [ WrapperStatisticalDistribution(self._rvX).fit(X1s) for X1s in lX1s ]
		
		## Find alpha-pooling parameters
		self._fit_pooling( rvY0 , lrvX0s )
		
		## Find 'future distribution of observations'
		self._rvY0  = self._pooling(lrvX0s)
		self._rvY1  = self._pooling(lrvX1s)
		
		## And set multiple quantile mappings
		self._qm0 = []
		for rvX0 in lrvX0:
			self._qm0.append( Univariate_QM() )
			self._qm0[-1].rvY0 = self._rvY0
			self._qm0[-1].rvX0 = rvX0
		self._qm1 = []
		for rvX1 in lrvX1:
			self._qm1.append( Univariate_QM() )
			self._qm1[-1].rvY0 = self._rvY1
			self._qm1[-1].rvX0 = rvX1
		
		return self
	##}}}
	
	def predict( self , *args ):##{{{
		
		## Check and split calibration / projection
		nmod = len(self._qm0)
		if len(args) == nmod:
			lX1 = args
			lX0 = None
		elif len(args) == 2 * nmod:
			lX1 = args[:nmod]
			lX0 = args[nmod:]
		else:
			raise ValueError( "Projection of each model must be model, and optionally calibration of each model" )
		
		## Correction
		lZ1 = []
		for X1,qm1 in zip(lX1,self._qm1):
			lZ1.append(qm1.predict(X1))
		lZ0 = None
		if lX0 is not None:
			lZ0 = []
			for X0,qm0 in zip(lX0,self._qm0):
				lZ0.append(qm0.predict(X0))
		
		##
		if lZ0 is not None:
			return lZ1+lZ0
		return lZ1
	##}}}
	
	## Properties ##{{{
	
	@property
	def alpha(self):
		return self._a
	
	@property
	def a(self):
		return self._a
	
	@a.setter
	def a( self , a ):
		self._a = a
		
		x = np.linspace( 0 , 1 , 1000 )
		y = self._gpool(x)
		self._igpool = sci.interp1d( y , x , bounds_error = False , fill_value = (0,1) )
	
	@property
	def w(self):
		return self._w
	
	@w.setter
	def w( self , w ):
		w = np.array([w]).ravel()
		self._w = w / w.sum()
	
	##}}}
	
##}}}

class AlphaPooling(MultiUBC):##{{{
	
	"""
	SBCK.mm.AlphaPooling
	====================
	AlphaPooling multi-model bias correction method (see [1]).
	
	Example
	-------
	>>> import SBCK
	>>>
	>>> ## Parameters
	>>> np.random.seed(42)
	>>> size    = 1_000
	>>> nmod    = 3
	>>> locs0   = np.linspace( -5 , 5 , nmod )
	>>> locs1   = np.linspace( 5 , 15 , nmod )
	>>> scales1 = np.linspace( 0.1 , 3 , nmod )
	>>> scales0 = np.linspace( 0.5 , 5 , nmod )
	>>> 
	>>> ## Data
	>>> Y0  =  np.random.normal( loc = 0 , scale = 1 , size = size )
	>>> lX0 = [np.random.normal( loc = locs0[i] , scale = scales0[i] , size = size ) for i in range(nmod)]
	>>> lX1 = [np.random.normal( loc = locs1[i] , scale = scales1[i] , size = size ) for i in range(nmod)]
	>>> 
	>>> ## mm correction, fix alpha
	>>> mm  = SBCK.mm.AlphaPooling( alpha = 3 ).fit( Y0 , *(lX0+lX1) )
	>>> lZ  = mm.predict( *(lX1+lX0) )
	>>> lZ1 = lZ[:nmod]
	>>> lZ0 = lZ[nmod:]
	>>> 
	>>> ## mm correction, fit alpha
	>>> mm  = SBCK.mm.AlphaPooling().fit( Y0 , *(lX0+lX1) )
	>>> lZ  = mm.predict( *(lX1+lX0) )
	>>> lZ1 = lZ[:nmod]
	>>> lZ0 = lZ[nmod:]
	```
	
	References
	----------
	[1] Vrac, M. and al: Distribution-based pooling for combination and
	multi-model bias correction of climate simulations, Earth Syst. Dynam., 15,
	735-762, doi:10.5194/esd-15-735-2024, 2024
	
	"""
	
	def __init__( self , alpha = None , rvY = rv_empirical , rvX = rv_empirical ):##{{{
		"""
		SBCK.mm.AlphaPooling.__init__
		=============================
		
		Arguments
		---------
		alpha: float | None
			The alpha parameter, if None, infered during the fit.
		rvY: SBCK.tools.<law> | scipy.stats.<law>
			Law of references
		rvX: SBCK.tools.<law> | scipy.stats.<law>
			Law of models
		"""
		
		## Build args for MultiUBC
		if not isinstance( alpha , (list,tuple) ):
			if isinstance( rvY , (list,tuple) ):
				alpha = [alpha for _ in range(len(rvY))]
			elif isinstance( rvX , (list,tuple) ):
				alpha = [alpha for _ in range(len(rvX))]
			else:
				alpha = [alpha]
		if not isinstance( rvY , (list,tuple) ):
			if isinstance( alpha , (list,tuple) ):
				rvY = [rvY for _ in range(len(alpha))]
			elif isinstance( rvX , (list,tuple) ):
				rvY = [rvY for _ in range(len(rvX))]
			else:
				rvY = [rvY]
		if not isinstance( rvX , (list,tuple) ):
			if isinstance( alpha , (list,tuple) ):
				rvX = [rvX for _ in range(len(alpha))]
			elif isinstance( rvY , (list,tuple) ):
				rvX = [rvX for _ in range(len(rvY))]
			else:
				rvX = [rvX]
		if not len(set({len(alpha),len(rvX),len(rvY)})) == 1:
			raise ValueError( f"Incoherent arguments between alpha, rvY and rvX" )
		args = [ (a,rvy,rvx) for a,rvy,rvx in zip(alpha,rvY,rvX) ]
		
		## And init upper class
		super().__init__( "AlphaPooling" , Univariate_AlphaPooling , args = args )
	##}}}
	
##}}}

