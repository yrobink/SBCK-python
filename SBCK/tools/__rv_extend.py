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
import scipy.stats as sc
import scipy.interpolate as sci
import scipy.optimize as sco

from .__stats import gpdfit
from ..tools.__sys import deprecated


##############
## rv_class ##
##############

def io_type(func):##{{{
	"""
	SBCK.tools.io_type
	==================
	
	Decorator of the cdf, icdf and pdf method of rv_base used to cast input
	data in numpy.ndarray, and re-cast to original type in output.
	
	"""
	def wrapper( self , x ):
		
		## Transform to 1d array
		xt = np.atleast_1d( np.asarray(x) ).astype(float).ravel()
		
		## Apply function
		yt = func( self , xt )
		
		## And go back to x
		if np.isscalar(x):
			return float(yt[0])
		
		y = x.copy().astype(float) + np.nan
		y[:] = yt.reshape(y.shape)
		
		return y
	
	return wrapper
##}}}

class rv_base:##{{{
	"""
	SBCK.tools.rv_base
	==================
	Base class of random variable, used to be derived.
	"""
	
	def __init__(self):##{{{
		pass
	##}}}
	
	def rvs( self , size ):##{{{
		return self.icdf( np.random.uniform( size = size ) )
	##}}}
	
	## _cdf, _icdf and _pdf ##{{{
	
	def _cdf( self , x ):
		raise NotImplementedError
	
	def _icdf( self , p ):
		raise NotImplementedError
	
	def _pdf( self , x ):
		raise NotImplementedError
	
	##}}}
	
	@io_type
	def cdf( self , x ):##{{{
		return self._cdf(x)
	##}}}
	
	@io_type
	def icdf( self , p ):##{{{
		return self._icdf(p)
	##}}}
	
	def sf( self , x ):##{{{
		return 1 - self.cdf(x)
	##}}}
	
	def isf( self , p ):##{{{
		return self.icdf(1-p)
	##}}}
	
	def ppf( self , p ):##{{{
		return self.icdf(p)
	##}}}
	
	@io_type
	def pdf( self , x ):##{{{
		return self._pdf(x)
	##}}}
	
	def _init_icdf_from_cdf( self , xmin , xmax ):##{{{
		
		x    = np.linspace( xmin , xmax , 1000 )
		p    = self._cdf(x)
		icdf = sci.interp1d( p , x , bounds_error = False , fill_value = (xmin,xmax) )
		
		return icdf
	##}}}
	
	## Properties ##{{{
	
	@property
	def a(self):
		return self._icdf(0)
	
	@property
	def b(self):
		return self._icdf(1)
	##}}}
	
##}}}

class rv_empirical(rv_base):##{{{
	"""
	SBCK.tools.rv_empirical
	=======================
	Empirical histogram class. The differences with scipy.stats.rv_histogram
	are 1. the fit method and 2. the way to infer the cdf and the icdf. Here:
	
	>>> X ## Input
	>>> rvX = rv_empirical( *rv_empirical.fit(X) )
	"""
	
	def __init__( self , cdf = None , icdf = None , pdf = None , *args , X = None , **kwargs ):##{{{
		"""
		SBCK.tools.rv_empirical.__init__
		================================
		
		Arguments
		---------
		cdf: callable | None
			The cumulative density function (cdf), infered if X is given
		icdf: callable | None
			The inverse of the cdf, infered if X is given
		pdf: callable | None
			The probability density function, infered if X is given
		X: np.ndarray | None
			Init the rv_empirical with X
		"""
		super().__init__()
		self._fcdf  = None
		self._ficdf = None
		self._fpdf  = None
		if cdf is not None and icdf is not None and pdf is not None:
			self._fcdf  = cdf
			self._ficdf = icdf
			self._fpdf  = pdf
		elif cdf is not None and icdf is not None and kwargs.get( "no_pdf" , False ):
			self._fcdf  = cdf
			self._ficdf = icdf
		elif X is not None:
			cdf,icdf,pdf = type(self).fit(X)
			self._fcdf  = cdf
			self._ficdf = icdf
			self._fpdf  = pdf
		else:
			raise ValueError( "Bad input arguments" )
	##}}}
	
	@staticmethod
	def fit( X , *args , **kwargs ):##{{{
		"""
		SBCK.tools.rv_empirical.fit
		===========================
		
		Fit method
		
		Arguments
		---------
		X: np.ndarray | None
			Init the rv_empirical with X
		kwargs:
			bin_number: int = int(0.1*X.size)
				Numbers of bin used to infer the pdf, must be between bin_min
				and bin_max
			bin_min: int = 21
				Minimal numbers of bin used to infer the pdf
			bin_max: int = 101
				Maximal numbers of bin used to infer the pdf
		
		
		Returns
		-------
		cdf: callable
			The cumulative density function (cdf), infered if X is given
		icdf: callable
			The inverse of the cdf, infered if X is given
		pdf: callable
			The probability density function, infered if X is given
		"""
		
		Xs = np.sort(X.squeeze())
		Xr = sc.rankdata(Xs,method="max")
		p  = np.unique(Xr) / X.size
		q  = Xs[np.unique(Xr)-1]
		
		if p[0] > 0:
			q  = np.hstack( (q[0] - np.sqrt(np.finfo(float).resolution),q) )
			p  = np.hstack( (0,p) )
		
		icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (q[0],q[-1]) )
		cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (0,1) )
		
		dq   = 0.05 * (q.max() - q.min())
		bins = np.linspace( q.min() - dq , q.max() + dq , min( max( kwargs.get("bin_number",int(0.1*X.size)) , kwargs.get("bin_min",21) ) , kwargs.get("bin_max",101) ) )
		h,c  = np.histogram( X , bins , density = True )
		c    = (c[1:] + c[:-1]) / 2
		pdf  = sci.interp1d( c , h , bounds_error = False , fill_value = (0,0) ) 
		
		return (cdf,icdf,pdf)
	##}}}
	
	## _cdf, _icdf and _pdf ##{{{
	
	def _cdf( self , x ):
		return self._fcdf(x)
	
	def _icdf( self , p ):
		return self._ficdf(p)
	
	def _pdf( self , x ):
		return self._fpdf(x)
	
	##}}}
	
	
##}}}

class rv_empirical_ratio(rv_empirical):##{{{
	"""
	SBCK.tools.rv_empirical_ratio
	=============================
	Extension of SBCK.tools.rv_empirical taking into account of a "ratio" part,
	i.e., instead of fitting:
	P( X < x )
	We fit separatly the frequency of 0 and:
	P( X < x | X > 0 )
	"""
	
	def __init__( self , cdf = None , icdf = None , pdf = None , p0 = None , *args , X = None , **kwargs ):##{{{
		"""
		SBCK.tools.rv_empirical.__init__
		================================
		
		Arguments
		---------
		cdf: callable | None
			The cumulative density function (cdf), infered if X is given
		icdf: callable | None
			The inverse of the cdf, infered if X is given
		pdf: callable | None
			The probability density function, infered if X is given
		p0: float | None
			The probability at 0
		X: np.ndarray | None
			Init the rv_empirical_ratio with X
		"""
		
		self._p0  = p0
		if cdf is not None and icdf is not None and pdf is not None and p0 is not None:
			super().__init__( cdf , icdf , pdf )
		elif cdf is not None and icdf is not None and p0 is not None and kwargs.get( "no_pdf" , False ):
			super().__init__( cdf , icdf , pdf , no_pdf = True )
		elif X is not None:
			cdf,icdf,pdf,p0 = type(self).fit(X)
			super().__init__( cdf , icdf , pdf )
			self._p0   = p0
		else:
			raise ValueError( "Bad input arguments" )
	##}}}
	
	## Properties ##{{{
	
	@property
	def p0(self):
		return self._p0
	
	##}}}
	
	@staticmethod
	def fit( X , *args , **kwargs ):##{{{
		"""
		SBCK.tools.rv_empirical_ratio.fit
		=================================
		
		Fit method
		
		Arguments
		---------
		X: np.ndarray | None
			Init the rv_empirical_ratio with X
		
		Returns
		-------
		cdf: callable
			The cumulative density function (cdf), infered if X is given
		icdf: callable
			The inverse of the cdf, infered if X is given
		pdf: callable
			The probability density function, infered if X is given
		p0: float
			The probability at 0
		"""
		Xp = X[X>0]
		p0 = 1 - np.sum( X>0 ) / X.size
		return rv_empirical.fit( Xp , *args , **kwargs) + (p0,)
	##}}}
	
	def _cdf( self , x ):##{{{
		p = np.zeros_like(x) + np.nan
		idxp = x > 0
		idxn = x < 0
		idx0 = ~idxp & ~idxn
		p[idxp] = (1-self.p0) * super()._cdf( x[idxp] ) + self.p0
		p[idx0] = self.p0
		p[idxn] = 0
		return p
	##}}}
	
	def _icdf( self , p ):##{{{
		
		x    = np.zeros_like(p) + np.nan
		idxp = p > self.p0
		idx0 = ~idxp
		x[idxp] = super()._icdf( (p[idxp] - self.p0) / (1-self.p0) )
		x[idx0] = 0
		return x
	##}}}
	
	def _pdf( self , x ):##{{{
		
		pdf = np.zeros_like(x) + np.nan
		xp  = x[x>0].min() / 2
		xn  = x[x<0].max() / 2
		
		idxp = x > xp
		idxn = x < xn
		idx0 = ~idxp & ~idxn
		
		pdf[idxn] = 0
		pdf[idxp] = super()._pdf(x[idxp]) * (1 - self.p0)
		pdf[idx0] = sc.uniform.pdf( x[idx0] , loc = xp , scale = xn ) * self.p0
		return pdf
	##}}}
	
##}}}

class rv_empirical_gpd(rv_empirical):##{{{
	"""
	SBCK.tools.rv_empirical_gpd
	===========================
	Empirical histogram class where tails are given by the fit of Generalized
	Pareto Distribution.
	
	"""
	
	def __init__( self , *args , X = None , p = 0.1 , **kwargs ):##{{{
		"""
		SBCK.tools.rv_empirical_gpd.__init__
		====================================
		
		Arguments
		---------
		args (if given):
			cdf: callable
				The cumulative density function (cdf), infered if X is given
			icdf: callable
				The inverse of the cdf, infered if X is given
			pdf: callable
				The probability density function, infered if X is given
			locl: float
				Location parameter of the left GPD
			locr: float
				Location parameter of the right GPD
			gpdl: scipy.stats.genpareto
				Law of the left tail
			gpdr: scipy.stats.genpareto
				Law of the right tail
			pl: float
				Probability of the left tail
			pr: float
				Probability of the right tail
		X: np.ndarray | None
			Init the rv_empirical_gpd with X
		p: float | tuple | list
			Probability of the tailS, i.e.
			 - if p is a float, left and right tails have a weight of p/2.
			 - if p is a tuple | list, then pl,pr = p
			Not used if pl + pr = p are given in args.
		"""
		
		if X is not None:
			cdf,icdf,pdf,locl,locr,gpdl,gpdr,pl,pr = rv_empirical_gpd.fit( X , p )
		else:
			if len(args) == 7: ## p is not given
				cdf,icdf,pdf,locl,locr,gpdl,gpdr = args
				if isinstance(p,float):
					pl = p / 2
					pr = 1 - p / 2
				else:
					pl,pr = p
			elif len(args) == 8: ## p given as a single value
				cdf,icdf,pdf,locl,locr,gpdl,gpdr,p = args
				pl = p / 2
				pr = 1 - p / 2
			elif len(args) == 9: ## pl and pr are given
				cdf,icdf,pdf,locl,locr,gpdl,gpdr,pl,pr = args
			else:
				raise ValueError("Incoherent arguments")
		
		super().__init__( cdf , icdf , pdf )
		self._locl = locl
		self._locr = locr
		self._gpdl = gpdl
		self._gpdr = gpdr
		self._pl   = pl
		self._pr   = pr
	##}}}
	
	@staticmethod
	def fit( X , p = 0.1 ):##{{{
		"""
		SBCK.tools.rv_empirical_gpd.fit
		===============================
		
		Arguments
		---------
		X: np.ndarray | None
			Init the rv_empirical_gpd with X
		p: float | tuple | list
			Probability of the tailS, i.e.
			 - if p is a float, left and right tails have a weight of p/2.
			 - if p is a tuple | list, then pl,pr = p
			Not used if pl + pr = p are given in args.
		"""
		
		## Find pl and pr
		if isinstance( p , float ):
			pl = p / 2
			pr = 1 - p / 2
		else:
			pl,pr = p
		
		## Location parameter
		locl = np.quantile( X , pl )
		locr = np.quantile( X , pr )
		
		## Cut X in the three part
		idxl = X < locl
		idxr = X > locr
		idxm = ~idxl & ~idxr
		Xl   = X[idxl] - locl
		Xr   = X[idxr] - locr
		Xm   = X[idxm]
		
		## Empirical part
		cdf,icdf,pdf = rv_empirical.fit(Xm)
		
		## GPD left fit
		scl,shl = gpdfit(-Xl)
		gpdl    = sc.genpareto( loc = 0 , scale = scl , c = shl )
		
		## GPD right fit
		scr,shr = gpdfit(Xr)
		gpdr    = sc.genpareto( loc = 0 , scale = scr , c = shr )
		
		return cdf,icdf,pdf,locl,locr,gpdl,gpdr,pl,pr
	##}}}
	
	def _pdf( self , x ):##{{{
		
		pdf = np.zeros_like(x)
		
		## Split index into left, middle and right part
		idxl = x < self.locl
		idxr = x > self.locr
		idxm = ~idxl & ~idxr
		
		## Empirical
		pdf[idxm] = rv_empirical._pdf( self , x[idxm] ) * (self.pr - self.pl)
		
		## Left part
		pdf[idxl] = self.pl * self._gpdl.pdf( -(x[idxl] - self.locl) )
		
		## Right part
		pdf[idxr] = self._gpdr.pdf(x[idxr] - self.locr) * (1-self.pr)
		
		return pdf
	##}}}
	
	def _cdf( self , x ):##{{{
		
		p = np.zeros_like(x) + np.nan
		
		## Split index into left, middle and right part
		idxl = x < self.locl
		idxr = x > self.locr
		idxm = ~idxl & ~idxr
		
		## Empirical
		p[idxm] = rv_empirical._cdf( self , x[idxm] ) * (self.pr - self.pl) + self.pl
		
		## Left part
		p[idxl] = self.pl * self._gpdl.sf( -(x[idxl] - self.locl) )
		
		## Right part
		p[idxr] = self.pr + (1-self.pr) * self._gpdr.cdf(x[idxr] - self.locr)
		
		return p
	##}}}
	
	def _icdf( self , p ):##{{{
		
		x = np.zeros_like(p) + np.nan
		
		## Split index into left, middle and right part
		idxl = p < self.pl
		idxr = p > self.pr
		idxm = ~idxl & ~idxr
		
		## Empirical part
		x[idxm] = rv_empirical._icdf( self , ( p[idxm] - self.pl ) / (self.pr - self.pl) )
		
		## Left part
		x[idxl] = self.locl - self._gpdl.isf( p[idxl] / self.pl )
		
		## Right part
		x[idxr] = self.locr + self._gpdr.ppf( (p[idxr] - self.pr) / (1-self.pr ))
		
		return x
	##}}}
	
	## Attributes ##{{{
	
	@property
	def pl(self):
		return self._pl
	
	@property
	def pr(self):
		return self._pr
	
	@property
	def locl(self):
		return self._locl
	
	@property
	def locr(self):
		return self._locr
	
	@property
	def scalel(self):
		return float(self._gpdl.kwds["scale"])
	
	@property
	def scaler(self):
		return float(self._gpdr.kwds["scale"])
	
	@property
	def shapel(self):
		return float(self._gpdl.kwds["c"])
	
	@property
	def shaper(self):
		return float(self._gpdr.kwds["c"])
	
	##}}}
	
##}}}

class rv_density(rv_base):##{{{
	"""
	SBCK.tools.rv_density
	=====================
	Empirical density class. Use a gaussian kernel. 'bw_method' argument
	can be given as kwargs to scipy.stats.gaussian_kde
	
	"""
	
	def __init__( self , *args , **kwargs ):##{{{
		"""
		SBCK.tools.rv_density.__init__
		==============================
		
		Arguments
		---------
		args (if given):
			kernel: scipy.stats.gaussian_kde
				Kernel of X
			xmin: float
				left bound of the support
			xmax: float
				right bound of the support
		kwargs (if given):
			X: np.ndarray
				Init the rv_density with X
			bw_method:
				bandwidth method, see scipy.stats.gaussian_kde
		"""
		super().__init__()
		self._kernel = None
		if kwargs.get("X") is not None:
			X = kwargs.get("X")
			self._kernel = sc.gaussian_kde( X.squeeze() , bw_method = kwargs.get("bw_method") )
			self._icdf = self._init_icdf_from_cdf( X.min() , X.max() )
		elif len(args) > 0:
			self._kernel = args[0]
			self._icdf = self._init_icdf_from_cdf( args[1] , args[2] )
		
	##}}}
	
	def fit( X , bw_method = None ):##{{{
		"""
		SBCK.tools.rv_density.fit
		=========================
		
		Arguments
		---------
		X: np.ndarray
			Init the rv_density with X
		bw_method:
			bandwidth method, see scipy.stats.gaussian_kde
		
		Returns
		-------
		kernel: scipy.stats.gaussian_kde
			Kernel of X
		xmin: float
			left bound of the support
		xmax: float
			right bound of the support
		"""
		kernel = sc.gaussian_kde( X , bw_method = bw_method )
		return (kernel,X.min(),X.max())
	##}}}
	
	def _cdf( self , x ):##{{{
		cdf = np.apply_along_axis( lambda z: self._kernel.integrate_box_1d( -np.inf , z ) , 1 , x.reshape(-1,1) )
		cdf[cdf < 0] = 0
		cdf[cdf > 1] = 1
		return cdf.squeeze()
	##}}}
	
	def _pdf( self , x ):##{{{
		return self._kernel.pdf(x)
	##}}}
	
##}}}

class rv_mixture(rv_base):##{{{
	"""
	SBCK.tools.rv_mixture
	=====================
	Mixture of distributions. A fit method is implemented, but not really work.
	
	"""
	
	def __init__( self , *args , **kwargs ):##{{{
		"""
		SBCK.tools.rv_mixture.__init__
		==============================
		
		Arguments
		---------
		args:
			A list of scipy.stats.* distribution (initialized), and a vector of
			weights.
		"""
		super().__init__()
		
		self._dist    = []
		self._weights = None
		for d in args:
			if isinstance( d , (list,tuple,np.ndarray) ):
				self._weights = np.array(d).ravel()
			else:
				self._dist.append(d)
		
		if self._weights is None:
			if "weights" in kwargs:
				self._weights = kwargs["weights"]
			else:
				self._weights = np.ones(len(self._dist))
		self._weights  = np.array([self._weights]).ravel()
		self._weights /= self._weights.sum()
		
		## Build the icdf
		xmin =  np.inf
		xmax = -np.inf
		for dist in self._dist:
			xmin = min( xmin , dist.ppf(  1e-3) )
			xmax = max( xmax , dist.ppf(1-1e-3) )
		self._icdf = self._init_icdf_from_cdf(xmin,xmax)
	##}}}
	
	def _pdf( self , x ):##{{{
		pdf = np.zeros_like(x)
		for i in range(len(self._dist)):
			pdf += self._dist[i].pdf(x) * self._weights[i]
		return pdf
	##}}}
	
	def _cdf( self , x ):##{{{
		cdf = np.zeros_like(x)
		for i in range(len(self._dist)):
			cdf += self._dist[i].cdf(x) * self._weights[i]
		return cdf
	##}}}
	
	@staticmethod
	def fit( X , *args ):##{{{
		"""
		SBCK.tools.rv_mixture.fit
		=========================
		
		Use a MLE method, but not really work
		
		Arguments
		---------
		X: np.ndarray
			Init the rv_density with X
		args:
			A list of scipy.stats.* distribution, not initialized
		
		Returns
		-------
		- A list of scipy.stats.* distribution, initialized
		- A vector of weights
		"""
		
		## Step 1, find the numbers of parameters
		nhpars = []
		x0     = []
		for law in args:
			ehpar = list(law.fit(X))
			x0 = x0 + ehpar
			nhpars.append( len(ehpar) )
		thpar = sum(nhpars)
		
		## Define the likelihood
		def mle( x , X , nhpars , *args ):
			
			## Extract hyper-parameters and weights
			thpar = sum(nhpars)
			hpars = x[:thpar]
			ws    = np.exp(x[thpar:])
			ws    = ws / ws.sum()
			
			## Split hpar in hpar for each law
			i0,i1  = 0,0
			lhpars = []
			for n in nhpars:
				i1 = i0 + n
				lhpars.append(hpars[i0:i1])
				i0 = i1
			
			## And compute likelihood
			ll = 0
			for law,hpar,w in zip(args,lhpars,ws):
				ll += w * law.logpdf( X , *hpar.tolist() ).sum()
			
			return -ll
		
		## Define the init point
		x0 = np.array( [x0 + np.zeros(len(args)).tolist()] ).ravel()
		
		## Optimization
		res = sco.minimize( mle , x0 , args = (X,nhpars) + args , method = "Nelder-Mead" )
		x   = res.x
		
		## Split output parameters
		hpars = x[:thpar]
		ws    = np.exp(x[thpar:])
		ws    = ws / ws.sum()
		i0,i1  = 0,0
		lhpars = []
		for n in nhpars:
			i1 = i0 + n
			lhpars.append(hpars[i0:i1])
			i0 = i1
		
		##
		oargs = []
		for law,hpar in zip(args,lhpars):
			oargs.append( law( *hpar.tolist() ) )
		oargs.append(ws)
		
		return tuple(oargs)
	##}}}
	
##}}}


###########################
## Multivariate rv_class ##
###########################

class mrv_base:##{{{
	"""
	SBCK.tools.mrv_base
	==================
	Class used to transform univariate rv in multivariate rv. Each margins is
	fitted separately.
	"""
	
	def __init__( self , *args ):
		self.ndim  = len(args)
		self._claw = args
		self._law  = []
	
	def fit( self , X ):
		if X.ndim == 1:
			X = X.reshape(-1,1)
		if self.ndim == 0:
			self.ndim  = X.shape[1]
			self._claw = [rv_empirical for _ in range(self.ndim)]
		elif not self.ndim == X.shape[1]:
			raise ValueError( "Dimensions of X not compatible with arguments" )
		for i in range(self.ndim):
			claw = self._claw[i]
			self._law.append( claw( *claw.fit( X[:,i] ) ) )
		
		return self
	
	def rvs( self , size = 1 ):
		return np.array( [ self._law[i].rvs(size=size) for i in range(self.ndim) ] ).T.copy()
	
	def cdf( self , x ):
		x = x.reshape(-1,self.ndim)
		return np.array( [ self._law[i].cdf(x[:,i]) for i in range(self.ndim) ] ).T.copy()
	
	def sf( self , x ):
		return 1 - self.cdf(x)
	
	def icdf( self , p ):
		p = p.reshape(-1,self.ndim)
		return np.array( [ self._law[i].ppf(p[:,i]) for i in range(self.ndim) ] ).T.copy()
	
	def isf( self , p ):
		return self.icdf(1-p)
	
	def ppf( self , p ):
		return self.icdf(p)
	
##}}}


############################
## Wrapper of QM and CDFt ##
############################

class WrapperStatisticalDistribution:##{{{
	
	def __init__( self , law = rv_empirical ):
		self.law = law
	
	def is_frozen( self ):
		return isinstance(self.law,sc._distn_infrastructure.rv_frozen)
	
	def is_parametric( self ):
		raise NotImplementedError
	
	def cdf( self , x ):
		return self.law.cdf(x)
	
	def icdf( self , p ):
		return self.law.ppf(p)
	
	def fit( self , X ):
		if not self.is_frozen():
			self.law = self.law( *self.law.fit( X.squeeze() ) )
		return self
##}}}


################
## Deprecated ##
################

@deprecated( "rv_histogram is renamed rv_empirical since the version 2.0.0" )
class rv_histogram(rv_empirical):##{{{
	def __init__( self , *args , **kwargs ):
		super().__init__(*args,**kwargs)
##}}}

@deprecated( "rv_ratio_histogram is renamed rv_empirical_ratio since the version 2.0.0" )
class rv_ratio_histogram(rv_empirical_ratio):##{{{
	def __init__( self , *args , **kwargs ):
		super().__init__(*args,**kwargs)
##}}}

@deprecated( "mrv_histogram is renamed mrv_base since the version 2.0.0" )
class mrv_histogram(mrv_base):##{{{
	def __init__( self , *args , **kwargs ):
		super().__init__(*args,**kwargs)
##}}}


