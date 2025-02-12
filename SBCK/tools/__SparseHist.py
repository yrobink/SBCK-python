# -*- coding: utf-8 -*-

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

from .__tools_cpp import SparseHist as SparseHistCPP
from .__stats import bin_width_estimator


#############
## Classes ##
#############

class SparseHist:##{{{
	"""
	SBCK.tools.SparseHist
	=====================
	Sparse Histogram class, interface to a c++ class.
	
	Attributes:
	- 'c': center of bins,
	- 'p': probability of bins,
	- 'shape': shape of c,
	- 'ndim': numbers of dimensions,
	- 'size': total size (product of shape),
	- 'sizep': number of bins
	
	"""
	
	
	def __init__( self , X , bin_width = None , bin_origin = None ):##{{{
		"""
		SBCK.tools.SparseHist.__init__
		==============================
		Sparse Histogram class, interface to a c++ class.
		
		Arguments
		---------
		
		X: np.ndarray
			Data to infer the histogram
		bin_width: np.ndarray
			Width of a bin
		bin_origin:
			left corner of one bin, default is 0,0,...
		
		"""
		
		## Check X
		if X.ndim == 1:
			X = X.reshape(-1,1)
		
		## Check bin_width and bin_origin
		if bin_width is None:
			bin_width = bin_width_estimator(X)
		if bin_origin is None:
			bin_origin = np.zeros_like(bin_width)
		
		## And init the SparseHist
		self._sparse_hist = SparseHistCPP( X , bin_width , bin_origin )
	##}}}
	
	def argwhere( self , X ):##{{{
		"""
		SBCK.tools.SparseHist.argwhere
		==============================
		Return the index of the bins in 'c' of the elements of X. So:
		x[:,i] is the bin defined by c[I[i],:]
		
		Arguments
		---------
		X: np.ndarray
			Data to infer the histogram
		
		Returns
		-------
		I: np.ndarray
			Index
		
		"""
		
		## Check X
		if X.ndim == 1:
			X = X.reshape(-1,1)
		size,ndim = X.shape
		
		## Finite index
		v    = np.isfinite(X).all(1)
		
		## Output
		I    = np.zeros(X.shape[0]) + self.shape[0] + 1
		I[v] = self._sparse_hist.argwhere(X[v,:])
		I    = np.where( I > -1 , I , self.shape[0] )
		
		return I.astype(int)
	##}}}
	
	## Properties ##{{{
	
	@property
	def c(self):
		return self._sparse_hist.c
	
	@property
	def p(self):
		return self._sparse_hist.p
	
	@property
	def sizep(self):
		return self._sparse_hist.p.size
	
	@property
	def shape(self):
		return self._sparse_hist.c.shape
	
	@property
	def ndim(self):
		return self._sparse_hist.c.ndim
	
	@property
	def size(self):
		return np.prod(self.shape)
	
	##}}}
	
##}}}


