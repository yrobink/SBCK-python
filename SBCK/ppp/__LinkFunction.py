
## Copyright(c) 2022 / 2025 Yoann Robin
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
from ..tools.__sys import deprecated


###########
## Class ##
###########

class LinkFunction(PrePostProcessing):##{{{
	"""
	SBCK.ppp.LinkFunction
	=====================
	
	This class is used to define pre/post processing class with a link function
	and its inverse. See also the PrePostProcessing documentation
	
	>>> ## Start with data
	>>> Y0,X0,X1 = SBCK.datasets.like_tas_pr(2000)
	>>> 
	>>> ## Define the link function
	>>> transform  = lambda x : x**3
	>>> itransform = lamnda x : x**(1/3)
	>>> 
	>>> ## And the PPP method
	>>> ppp = SBCK.ppp.LinkFunction( bc_method = SBCK.CDFt ,
	>>>                                transform_ = transform ,
	>>>                               itransform_ = itransform )
	>>> 
	>>> ## And now the correction
	>>> ## Bias correction
	>>> ppp.fit(Y0,X0,X1)
	>>> Z = ppp.predict(X1,X0)
	
	"""
	def __init__( self , *args , name = "LinkFunction" , transform_ = None , itransform_ = None , cols = None , **kwargs ):##{{{
		"""
		Constructor
		===========
		
		Arguments
		---------
		transform_: [callable]
			Function to transform the data
		itransform_: [callable]
			Function to inverse the transform of the data
		cols: [int or array of int]
			The columns to apply the SSR
		isaved: str
			Choose the threshold used for inverse transform. Can be "Y0", "X0"
			or "X1"
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		PrePostProcessing.__init__( self , *args , **kwargs )
		self._name         = name
		self._f_transform  = transform_
		self._f_itransform = itransform_
		self._cols = cols
		if cols is not None:
			self._cols = np.array( [cols] , dtype = int ).squeeze()
	
	##}}}
	
	## Transform and itransform functions ##{{{
	
	def _transform( self , X ):
		return self._f_transform(X)
	
	def _itransform( self , Xt ):
		return self._f_itransform(Xt)
	
	def transform( self , X ):
		"""
		Apply the transform
		"""
		if self._cols is None:
			return self._transform(X)
		Xt = X.copy()
		Xt[:,self._cols] = self._transform(X[:,self._cols])
		return Xt
	
	def itransform( self , Xt ):
		"""
		Apply the inverse transform
		"""
		if self._cols is None:
			return self._itransform(Xt)
		X = Xt.copy()
		X[:,self._cols] = self._itransform(Xt[:,self._cols])
		return X
	##}}}
	
##}}}

class LFAdd(LinkFunction):##{{{
	"""
	SBCK.ppp.LFAdd
	==============
	
	Addition link transform.
	
	"""
	def __init__( self , m : float , *args , cols = None , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		m : [float]
			The value to add
		cols: [int or array of int]
			The columns to apply the Link function
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		
		transform  = lambda x: x + m
		itransform = lambda x: x - m
		LinkFunction.__init__( self , *args , name = "LFAdd" , transform_ = transform , itransform_ = itransform , cols = cols , **kwargs )
##}}}

class LFMult(LinkFunction):##{{{
	"""
	SBCK.ppp.LFMult
	===============
	
	Multiplication link transform.
	
	"""
	def __init__( self , s : float , *args , cols = None , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		 : [float]
			The value used to multiply
		cols: [int or array of int]
			The columns to apply the Link function
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		
		transform  = lambda x: x * s
		itransform = lambda x: x / s
		LinkFunction.__init__( self , *args , name = "LFMult" , transform_ = transform , itransform_ = itransform , cols = cols , **kwargs )
##}}}

class LFMax(LinkFunction):##{{{
	"""
	SBCK.ppp.LFMax
	==============
	
	Max link.
	
	"""
	def __init__( self , M : float , *args , cols = None , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		M : [float]
			The max
		cols: [int or array of int]
			The columns to apply the Link function
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		
		transform  = lambda x: np.where( (x < M) | ~np.isfinite(x) , x , M )
		itransform = lambda x: np.where( (x < M) | ~np.isfinite(x) , x , M )
		LinkFunction.__init__( self , *args , name = "LFMax" , transform_ = transform , itransform_ = itransform , cols = cols , **kwargs )
##}}}

class LFMin(LinkFunction):##{{{
	"""
	SBCK.ppp.LFMin
	==============
	
	Min link.
	
	"""
	def __init__( self , M : float , *args , cols = None , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		M : [float]
			The min
		cols: [int or array of int]
			The columns to apply the Link function
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		
		transform  = lambda x: np.where( (x > M) | ~np.isfinite(x) , x , M )
		itransform = lambda x: np.where( (x > M) | ~np.isfinite(x) , x , M )
		LinkFunction.__init__( self , *args , name = "LFMin" , transform_ = transform , itransform_ = itransform , cols = cols , **kwargs )
##}}}

class LFSquare(LinkFunction):##{{{
	"""
	SBCK.ppp.LFSquare
	=================
	
	Square link transform, i.e.:
	- transform is given by lambda x: x**2
	- inverse transform is given by lambda x: sign(x) * sqrt(abs(x))
	
	"""
	
	def __init__( self , *args , cols = None , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		cols: [int or array of int]
			The columns to apply the Link function
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		transform  = lambda x : x**2
		itransform = lambda x : np.where( x > 0 , np.sqrt(np.abs(x)) , - np.sqrt(np.abs(x)))
		LinkFunction.__init__( self , *args , name = "LFSquare" , transform_ = transform , itransform_ = itransform , cols = cols , **kwargs )
##}}}

class LFLoglin(LinkFunction):##{{{
	"""
	SBCK.ppp.LFLoglin
	=================
	
	Log linear link transform, i.e.:
	- transform is given by s*log(x/s) + s if 0 < x < s, else x
	- inverse transform is given by s*exp( (x-s) / s ) if x < s, else x
	
	"""
	def __init__( self , *args , s = 1e-5 , cols = None , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		cols: [int or array of int]
			The columns to apply the Link function
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		if not s > 0:
			raise Exception( f"Parameter s = {s} must be non negative!" )
		self.s = s
		transform  = lambda x: np.where( (0 < x) & (x < s) , s * np.log( np.where( x > 0 , x , np.nan ) / s ) + s , np.where( x < 0 , np.nan , x ) )
		itransform = lambda x: np.where( x < s , s * np.exp( (x-s) / s ) , x )
		LinkFunction.__init__( self , *args , name = "LFLoglin" , transform_ = transform , itransform_ = itransform , cols = cols , **kwargs )
	
##}}}

class LFArctan(LinkFunction):##{{{
	"""
	SBCK.ppp.LFArctan
	=================
	
	Arctan link transform, to bound the correction between two values.
	
	"""
	def __init__( self , ymin : float , ymax : float , *args , cols = None , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		ymin : [float]
			Minimum
		ymax : [float]
			Maximum
		cols: [int or array of int]
			The columns to apply the Link function
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		
		f = (ymax - ymin) / np.pi
		transform  = lambda x: np.where( (x > ymin) & (x < ymax) , f * np.tan( (x - ymin) / f - np.pi / 2 ) , np.nan )
		itransform = lambda x: (np.pi / 2 + np.arctan(x/f) ) * f + ymin
		LinkFunction.__init__( self , *args , name = "LFArctan" , transform_ = transform , itransform_ = itransform , cols = cols , **kwargs )
##}}}

class LFLogistic(LinkFunction):##{{{
	"""
	SBCK.ppp.LFLogistic
	===================
	
	Logistic link transform, to bound the correction between two values.
	Starting from a dataset bounded between ymin and ymax, the transform maps
	the interval [ymin,ymax] to R with:
	
	transform : x |-> - np.log( (ymax - ymin) / (x - ymin) - 1 ) / s
	
	and the inverse transform is the logistic function:
	
	itransform : y |-> (ymax - ymin) / ( 1 + np.exp(-s*y) ) + ymin
	
	
	
	"""
	def __init__( self , ymin : float , ymax : float , *args , s : float = 1 , tol : float = 1e-8 , cols = None , **kwargs ):
		"""
		Constructor
		===========
		
		Arguments
		---------
		ymin : [float]
			Minimum
		ymax : [float]
			Maximum
		s : [float]
			The slope around 0 of the transform, default to 1
		cols: [int or array of int]
			The columns to apply the Link function
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		
		self.ymin = ymin
		self.ymax = ymax
		self.s    = s
		self._tol = tol
		
		LinkFunction.__init__( self , *args , name = "LFLogistic" , cols = cols , **kwargs )
	
	def _transform( self , x ):
		xt  = x.copy()
		xt  = np.where( xt < self.ymax - self._tol , xt , self.ymax - self._tol )
		xt  = np.where( xt > self.ymin + self._tol , xt , self.ymin + self._tol )
		y   = - np.log( (self.ymax - self.ymin) / (xt - self.ymin) - 1 ) / self.s
		ivl = (x < self.ymin) | (x > self.ymax)
		y[ivl] = np.nan
		return y
	
	def _itransform( self , y ):
		x = y.copy()
		x = (self.ymax - self.ymin) / ( 1 + np.exp(-self.s*x) ) + self.ymin
		x = np.where( x < self.ymax - self._tol , x , self.ymax )
		x = np.where( x > self.ymin + self._tol , x , self.ymin )
		return x
	
##}}}


######################
## Deprecated names ##
######################

@deprecated( "PPPLinkFunction is renamed LinkFunction since the version 2.0.0" )
class PPPLinkFunction(LinkFunction):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPLinkFunction"
	##}}}
	
##}}}

@deprecated( "PPPAddLink is renamed LFAdd since the version 2.0.0" )
class PPPAddLink(LFAdd):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPAddLink"
	##}}}
	
##}}}

@deprecated( "PPPMultLink is renamed LFMult since the version 2.0.0" )
class PPPMultLink(LFMult):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPMultLink"
	##}}}
	
##}}}

@deprecated( "PPPMaxLink is renamed LFMax since the version 2.0.0" )
class PPPMaxLink(LFMax):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPMaxLink"
	##}}}
	
##}}}

@deprecated( "PPPMinLink is renamed LFMin since the version 2.0.0" )
class PPPMinLink(LFMin):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPMinLink"
	##}}}
	
##}}}

@deprecated( "PPPSquareLink is renamed LFSquare since the version 2.0.0" )
class PPPSquareLink(LFSquare):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPSquareLink"
	##}}}
	
##}}}

@deprecated( "PPPLogLinLink is renamed LFLoglin since the version 2.0.0" )
class PPPLogLinLink(LFLoglin):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPLogLinLink"
	##}}}
	
##}}}

@deprecated( "PPPArctanLink is renamed LFArctan since the version 2.0.0" )
class PPPArctanLink(LFArctan):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPArctanLink"
	##}}}
	
##}}}

@deprecated( "PPPLogisticLink is renamed LFLogistic since the version 2.0.0" )
class PPPLogisticLink(LFLogistic):##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		super().__init__( *args , **kwargs )
		self._name = "PPPLogisticLink"
	##}}}
	
##}}}

