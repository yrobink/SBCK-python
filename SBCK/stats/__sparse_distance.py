
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
import scipy.spatial.distance as ssd

from .__SparseHist import bin_width_estimator
from .__SparseHist import SparseHist
from ..tools.__OT import POTemd


##############
## Function ##
##############

def _to_SparseHist( func ):##{{{
	"""Decorators for distances functions to transform input in SparseHist
	"""
	def wrapper( muX: SparseHist | np.ndarray , muY: SparseHist | np.ndarray , *args , **kwargs ) -> float:
		
		if not isinstance(muX,SparseHist) and isinstance(muY,SparseHist):
			muXX = SparseHist( muX , muY.bin_width )
			return func( muXX , muY , **kwargs )
		elif isinstance(muX,SparseHist) and not isinstance(muY,SparseHist):
			muYY = SparseHist( muY , muX.bin_width )
			return func( muX , muYY , **kwargs )
		elif not isinstance(muX,SparseHist) and not isinstance(muY,SparseHist):
			bw = bin_width_estimator( muX , muY ).squeeze()
			muXX = SparseHist( muX , bw )
			muYY = SparseHist( muY , bw )
			return func( muXX , muYY , **kwargs )
		
		return func( muX , muY , **kwargs )
	return wrapper

##}}}

@_to_SparseHist
def chebyshev( muX: SparseHist , muY: SparseHist , normalized: bool = True ) -> float:##{{{
	"""
	Description
	===========
	Chebyshev distance between SparseHist, defines by
	dist = max_{ij} |x_i-y_i|

	Parameters
	----------
	muX      : SBCK.SparseHist or np.array
		Histogram or dataset
	muY      : SBCK.SparseHist or np.array
		Histogram or dataset
	normalized: bool
		If true, return maximal difference normalized by the maximal
		probability in a bins
	Return
	------
	cost   : float
		Minkowski distance between muX and muY
	"""
	dist = 0
	indx = muY.argwhere( muX.c )
	indy = muX.argwhere( muY.c )

	## Common elements of muX in muY
	g = np.argwhere( indx < muY.sizep ).ravel()
	for i in g:
		dist = max( dist , np.abs( muX.p[i] - muY.p[indx[i]] ) )
	
	## Elements of muX not in muY
	g = np.argwhere(indx == muY.sizep).ravel()
	for i in g:
		dist = max( dist , muX.p[i] )

	## Elements of muY not in muX
	g = np.argwhere(indy == muX.sizep).ravel()
	for i in g:
		dist = max( dist , muY.p[i] )
	
	if normalized:
		dist = dist / max([muX.p.max(),muY.p.max()])

	return dist
##}}}

@_to_SparseHist
def energy( muX: SparseHist , muY: SparseHist , p: float = 2. , metric: str = "euclidean" ) -> float:##{{{
	"""
	Description
	===========
	Energy distance between sparse histograms

	Parameters
	----------
	muX      : SBCK.SparseHist or np.array
		Histogram or dataset
	muY      : SBCK.SparseHist or np.array
		Histogram or dataset
	p        : float
		Power of the metric function
	metric   : str or callable
		See scipy.spatial.distance.pdist
	
	Return
	------
	distance : float
		Estimation of energy distance
	"""
	sizeX = muX.sizep
	sizeY = muY.sizep
	XY = np.power( ssd.cdist( muX.c , muY.c , metric = metric ) , p ) * np.dot( muX.p.reshape( (sizeX,1) ) , muY.p.reshape( (1,sizeY) ) )
	XX = ssd.squareform( np.power( ssd.pdist( muX.c , metric = metric ) , p ) ) * np.dot( muX.p.reshape( (sizeX,1) ) , muX.p.reshape( (1,sizeX) ) )
	YY = ssd.squareform( np.power( ssd.pdist( muY.c , metric = metric ) , p ) ) * np.dot( muY.p.reshape( (sizeY,1) ) , muY.p.reshape( (1,sizeY) ) )

	return np.power( 2 * np.sum(XY) - np.sum(XX) - np.sum(YY) , 1. / p )
##}}}

@_to_SparseHist
def minkowski( muX: SparseHist , muY: SparseHist , p: float ) -> float:##{{{
	"""
	Description
	===========
	Minkowski distance between SparseHist, defines by
	dist^p = sum_{ij} |x_i-y_i|^p

	Parameters
	----------
	muX      : SBCK.SparseHist or np.array
		Histogram or dataset
	muY      : SBCK.SparseHist or np.array
		Histogram or dataset
	p      : float or np.inf (for Chebyshev distance)
		Power of the distance. If p = 2, it is euclidean distance

	Return
	------
	cost   : float
		Minkowski distance between muX and muY
	"""
	if p == np.inf:
		return chebyshev(muX,muY)
	
	dist = 0
	indx = muY.argwhere( muX.c )
	indy = muX.argwhere( muY.c )

	## Common elements of muX in muY
	ii = np.argwhere( indx > muY.sizep ).ravel()
	dist += np.sum( np.power( np.abs( muX.p[ii] - muY.p[indx[ii]] ) , p ) )
	
	## Elements of muX not in muY
	dist += np.sum( np.power( np.abs( muX.p[ np.argwhere(indx == muY.sizep).ravel() ] ) , p ) )
	
	## Elements of muY not in muX
	dist += np.sum( np.power( np.abs( muY.p[ np.argwhere(indy == muY.sizep).ravel() ] ) , p ) )
	
	return np.power( dist , 1. / p )
##}}}

@_to_SparseHist
def euclidean( muX: SparseHist , muY: SparseHist ) -> float:##{{{
	"""
	Description
	===========
	Euclidean distance between SparseHist

	Parameters
	----------
	muX      : SBCK.SparseHist or np.array
		Histogram or dataset
	muY      : SBCK.SparseHist or np.array
		Histogram or dataset

	Return
	------
	cost   : float
		Euclidean distance between muX and muY
	"""

	return minkowski( muX , muY , p = 2. )
##}}}

@_to_SparseHist
def manhattan( muX: SparseHist , muY: SparseHist ) -> float:##{{{
	"""
	Description
	===========
	Manhattan distance between SparseHist
	
	Parameters
	----------
	muX      : SBCK.SparseHist or np.array
		Histogram or dataset
	muY      : SBCK.SparseHist or np.array
		Histogram or dataset

	Return
	------
	cost   : float
		Manhattan distance between muX and muY
	"""

	return minkowski( muX , muY , p = 1. )
##}}}

@_to_SparseHist
def wasserstein( muX: SparseHist , muY: SparseHist , p: float = 2. , ot: POTemd = POTemd() , metric: str = "euclidean" ) -> float:##{{{
	"""
	Description
	===========
	Compute the Wasserstein metric between two sparse histograms. If ot is a Sinkhorn algorithm, the dissimilarity is returned.
	
	Parameters
	----------
	muX      : SBCK.SparseHist or np.array
		Histogram or dataset
	muY      : SBCK.SparseHist or np.array
		Histogram or dataset
	p      : float
		Power of the cost function
	metric : str or callable
		See scipy.spatial.distance.pdist
	
	Return
	------
	cost   : float
		Wasserstein cost between muX and muY
	
	References
	----------
	[1] Wasserstein, L. N. (1969). Markov processes over denumerable products of spaces describing large systems of automata. Problems of Information Transmission, 5(3), 47-52.
	
	"""
	
	cost = lambda OT : np.power( np.sum(OT.P * OT.C) , 1. / p )
	
	ot.power = p 
	ot.fit( muX , muY )
	if isinstance( ot , POTemd ):
		w = cost(ot)
		if not ot.state:
			w = np.nan
		if not abs(ot.P.sum() - 1) < 1e-6:
			w = np.nan
		return w
	else:
		costXY = cost(ot)
		ot.fit( muX , muX )
		costXX = cost(ot)
		ot.fit( muY , muY )
		costYY = cost(ot)
		return costXY - (costXX + costYY) / 2
##}}}


