
## Copyright(c) 2022 Yoann Robin
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


###########
## Class ##
###########

class PPPPreserveOrder(PrePostProcessing):##{{{
	"""
	SBCK.ppp.PPPPreserveOrder
	=========================
	
	The inverse transform of this PPP sort the data of the column 'cols' along
	rows. It is useful for example when tas, tasmin and tasmax are correted to
	ensure their order.
	
	"""

	def __init__( self , *args , cols = None , **kwargs ):##{{{
		PrePostProcessing.__init__( self , *args , **kwargs )
		
		self._cols = cols
		if cols is not None:
			self._cols = np.array( [cols] , dtype = int ).squeeze()
	##}}}
	
	def transform( self , X ):##{{{
		"""
    	Apply the PerserveOrder transform, in fact just identity
		"""
		return X
	##}}}
	
	def itransform( self , Xt ):##{{{
		"""
    	Apply the inverse PerserveOrder transform, i.e. sort along cols
		"""
		
		if self._cols is None:
			return Xt
		
		X = Xt.copy()
		X[:,self._cols] = np.sort( X[:,self._cols] , axis = 1 )
		
		return X
	##}}}
##}}}

class PPPDiffRef(PrePostProcessing): ##{{{
	"""
	SBCK.ppp.PPPDiffRef
	===================
	
	Transform a dataset such that all `lower` dimensions are replaced by
	the `ref` dimension minus the `lower`; and all `upper` dimensions are
	replaced by `upper` minus `ref`.
	
	>>> ## Start with data
	>>> X    = np.random.normal(size = size).reshape(-1,1)
	>>> sign = np.random.choice( [-1,1] , nfeat - 1 , replace = True )
	>>> for s in sign:
	>>> 	X = np.concatenate( (X,X[:,0].reshape(-1,1) + s * np.abs(np.random.normal( size = (size,1) ))) , -1 )
	>>> 
	>>> ## Define the PPP method
	>>> ref   = 0
	>>> lower = np.argwhere( sign == -1 ).ravel() + 1
	>>> upper = np.argwhere( sign ==  1 ).ravel() + 1
	>>> pppdr = SBCK.ppp.PPPDiffRef( ref , lower , upper )
	>>> 
	>>> ## And now change the dimension, and reverse the operation
	>>> Xt  = pppdr.transform(X)
	>>> Xit = pppdr.itransform(Xt)
	>>> 
	>>> print( np.max( np.abs( X - Xit ) ) ) ## == 0
	"""
	
	def __init__( self , ref , *args , lower = None , upper = None , **kwargs ): ##{{{
		"""
		Constructor
		===========
		
		Arguments
		---------
		ref: [int]
			The reference dimension
		lower: None or array of int
			Dimensions lower than ref
		upper: None or array of int
			Dimensions upper than ref
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		PrePostProcessing.__init__( self , *args , **kwargs )
		
		self.ref   = int(np.array([ref]).ravel()[0])
		self.lower = lower
		self.upper = upper
		
		if lower is not None and len(lower) == 0:
			self.lower = None
		if upper is not None and len(upper) == 0:
			self.upper = None
		
	##}}}
	
	def transform( self , X ):##{{{
		"""
    	Apply the DiffReff transform.
		"""
		
		Xt = X.copy()
		
		if self.lower is not None:
			for i in self.lower:
				Xt[:,i] = X[:,self.ref] - X[:,i]
		
		if self.upper is not None:
			for i in self.upper:
				Xt[:,i] = X[:,i] - X[:,self.ref]
		
		
		return Xt
	##}}}
	
	def itransform( self , Xt ):##{{{
		"""
    	Apply the inverse DiffRef transform.
		"""
		
		X = Xt.copy()
		
		if self.lower is not None:
			for i in self.lower:
				X[:,i] = Xt[:,self.ref] - Xt[:,i]
		
		if self.upper is not None:
			for i in self.upper:
				X[:,i] = Xt[:,i] + Xt[:,self.ref]
		
		return X
		
		##}}}
	
##}}}

class PPPDiffColumns(PrePostProcessing):##{{{
	"""
	SBCK.ppp.PPPDiffColumns
	=======================
	
	Similar to SBCK.ppp.PPPDiffRef, but diff columns are replaced by diff column
	minus ref columns (or reverse if sign = -1)
	
	"""
	
	def __init__( self , ref , diff , *args , sign = 1 , **kwargs ):##{{{
		"""
		Constructor
		===========
		
		Arguments
		---------
		ref: [list]
			The reference dimension
		diff: [list]
			The difference dimension
		sign: [list]
			The if upper of lower
		*args:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		*kwargs:
			All others arguments are passed to SBCK.ppp.PrePostProcessing
		"""
		
		PrePostProcessing.__init__( self , *args , **kwargs )
		
		self.ref  = ref
		self.diff = diff
		self.sign = 1 if sign > 0 else -1
		
	##}}}
	
	def transform( self , X ):##{{{
		"""
		Apply the transform
		"""
		Xt = X.copy()
		
		Xt[:,self.diff] = self.sign * (X[:,self.diff] - X[:,self.ref])
		
		return Xt
	##}}}
	
	def itransform( self , Xt ):##{{{
		"""
		Apply the inverse transform
		"""
		X = Xt.copy()
		
		X[:,self.diff] = X[:,self.ref] + self.sign * X[:,self.diff]
		
		return X
	##}}}
	
##}}}


