# -*- coding: utf-8 -*-

## Copyright(c) 2021 / 2024 Yoann Robin
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

import itertools as itt
import numpy as np
import scipy.stats as sc


###############
## Functions ##
###############

def yearly_window( ybeg_ , yend_ , wleft , wpred , wright , tleft_ , tright_ ):##{{{
	"""
	SBCK.tools.yearly_window
	========================
	Generator to iterate over years between ybeg_ and yend_, with a fitting
	window of lenght wleft + wpred + wright, and a centered predict window of
	length wpred.
	
	Arguments
	---------
	ybeg_:
		Starting year
	yend_:
		Ending year
	wleft:
		Lenght of left window
	wpred:
		Lenght of middle / predict window
	wright:
		Lenght of right window
	tleft_:
		Left bound
	tright_:
		Right bound
	
	Returns
	-------
	The generator
	
	Examples
	--------
	>>> ybeg_,yend_        = 2006,2100
	>>> wleft,wpred,wright = 5,10,5
	>>> tleft_,tright_     = 1951,2100
	>>> print( f"Iterate over {wleft}-{wpred}-{wright} window" )
	>>> print( " * L-bound / Fit-left / Predict-left / Predict-right / Fit-right / R-Bound" )
	>>> for tf0,tp0,tp1,tf1 in SBCK.tools.yearly_window( ybeg_ , yend_ , wleft, wpred , wright , tleft_ , tright_ ):
	>>> 	print( f" *    {tleft_} /     {tf0} /         {tp0} /          {tp1} /      {tf1} /    {tright_}" )
	>>>
	>>> ## Output
	>>> ## Iterate over 5-10-5 window
	>>> ##  * L-bound / Fit-left / Predict-left / Predict-right / Fit-right / R-Bound
	>>> ##  *    1951 /     2001 /         2006 /          2015 /      2020 /    2100
	>>> ##  *    1951 /     2011 /         2016 /          2025 /      2030 /    2100
	>>> ##  *    1951 /     2021 /         2026 /          2035 /      2040 /    2100
	>>> ##  *    1951 /     2031 /         2036 /          2045 /      2050 /    2100
	>>> ##  *    1951 /     2041 /         2046 /          2055 /      2060 /    2100
	>>> ##  *    1951 /     2051 /         2056 /          2065 /      2070 /    2100
	>>> ##  *    1951 /     2061 /         2066 /          2075 /      2080 /    2100
	>>> ##  *    1951 /     2071 /         2076 /          2085 /      2090 /    2100
	>>> ##  *    1951 /     2081 /         2086 /          2095 /      2100 /    2100
	>>> ##  *    1951 /     2081 /         2096 /          2100 /      2100 /    2100
	"""
	
	ybeg = int(ybeg_)
	yend = int(yend_)
	tleft  = int(tleft_)
	tright = int(tright_)
	
	tp0  = int(ybeg)
	tp1  = tp0 + wpred - 1
	tf0  = tp0 - wleft
	tf1  = tp1 + wright
	
	while not tp0 > yend:
		
		## Work on a copy, original used for iteration
		rtf0,rtp0,rtp1,rtf1 = tf0,tp0,tp1,tf1
		
		## Correction when the left window is lower than tleft
		if rtf0 < tleft:
			rtf1 = rtf1 + tleft - rtf0
			rtf0 = tleft
		
		## Correction when the right window is upper than yend
		if rtf1 > tright:
			rtf1 = tright
			rtf0 = rtf0 - (tf1 - tright)
		if rtp1 > tright:
			rtp1 = tright
		
		## The return
		yield [str(x) for x in [rtf0,rtp0,rtp1,rtf1]]
		
		## And iteration
		tp0 = tp1 + 1
		tp1 = tp0 + wpred - 1
		tf0 = tp0 - wleft
		tf1 = tp1 + wright
##}}}


#############
## Classes ##
#############

class Shift:##{{{
	"""
	SBCK.tools.Shift
	================
	
	Description
	-----------
	Shift class used to to transform a dataset X to [X[0:size-lag],X[1:size-lag+1],...]
	"""
	
	def __init__( self , lag , method = "row" , ref = 0 ):##{{{
		"""
		Initialisation of shift class
		
		Parameters
		----------
		lag    : integer
			Time lag of the shift
		method : string
			Inverse method, "row" or "col"
		ref    : integer
			Reference columns / rows to inverse
		"""
		self.lag = lag
		self.ref = ref
		self.method = method
	##}}}
	
	@property
	def ref( self ):
		return self._ref
	
	@ref.setter
	def ref( self , ref ):
		self._ref = ref % ( self.lag + 1 )
	
	@property
	def method( self ):
		return self._method
	
	@method.setter
	def method( self , _method ):
		self._method = _method if _method == "row" else "col"
	
	def transform( self , X ):##{{{
		"""
		Transform X to the shifted Xs with lag
		
		Parameters
		----------
		X : np.array
			dataset to shift
		
		Returns
		-------
		Xs: np.array
			dataset shifted
		"""
		if X.ndim == 1: X = X.reshape(-1,1) ## genericity to always have a matrix X.
		n_samples,n_features = X.shape
		Xs = np.zeros( ( n_samples - self.lag , ( self.lag + 1 ) * n_features ) )
		
		for i in range(self.lag+1):
			db = i * n_features
			de = i * n_features + n_features
			tb = i
			te = n_samples - ( self.lag + 1 ) + i + 1
			Xs[:,db:de] = X[tb:te,:]
		return Xs
	##}}}
	
	def _inverse_by_row( self , Xs ):##{{{
		n_features = int( Xs.shape[1] / ( self.lag + 1 ))
		n_samples  = Xs.shape[0] + self.lag
		
		Xi = np.zeros( (n_samples,n_features) )
		for r in itt.chain(range(self.lag+1),[self.ref]):
			idx  = np.arange( r , n_samples - self.lag , self.lag )
			Xs0  = Xs[idx[:-1],:-n_features].reshape(-1,n_features) ## Without last index, because the last is also the first of next
			Xs1  = Xs[idx[-1],:].reshape(-1,n_features)
			Xs01 = np.vstack( (Xs0,Xs1) )
			n_samples_01 = Xs01.shape[0]
			Xi[r:(r+n_samples_01),:] = Xs01
		
		return Xi
	##}}}
	
	def _inverse_by_col( self , Xs ):##{{{
		n_features = int( Xs.shape[1] / (self.lag + 1) )
		n_samples = Xs.shape[0] + self.lag
		Xu   = np.zeros( (n_samples,n_features) )
		
		for i in itt.chain(range(self.lag+1),[self.ref]):
			db = i * n_features
			de = i * n_features + n_features
			tb = i
			te = n_samples - ( self.lag + 1 ) + i + 1
			Xu[tb:te,:] = Xs[:,db:de]
		return Xu
	##}}}
	
	def inverse( self , Xs , method = None ):##{{{
		"""
		Inverse transform
		
		Parameters
		----------
		Xs  : np.array
			dataset to unshift
		
		Returns
		-------
		X: np.array
			dataset unshifted
		"""
		if method is not None: self.method = method
		if self.method == "col":
			return self._inverse_by_col(Xs)
		else:
			return self._inverse_by_row( Xs )
	##}}}
	
##}}}

class SlopeStoppingCriteria:##{{{
	def __init__( self , minit , maxit , tol ):
		self.minit    = minit
		self.maxit    = maxit
		self.nit      = -1
		self.tol      = tol
		self.stop     = False
		self.criteria = list()
		self.slope    = list()
	
	def initialize(self):
		self.nit      = -1
		self.stop     = False
		self.criteria = list()
		self.slope    = list()
	
	def append( self , value ):
		self.criteria.append(value)
		if self.nit > self.minit:
			slope,_,_,_,_ = sc.linregress( range(len(self.criteria)) , self.criteria )
			self.stop = np.abs(slope) < self.tol
			self.slope.append(slope)
	
	def __iter__(self):
		return self
	
	def __next__(self):
		self.nit += 1
		if not self.nit < self.maxit-1:
			self.stop = True
		if not self.stop:
			return self.nit
		raise StopIteration
##}}}

