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
import scipy.stats as sc
import scipy.special as scs


###############
## Functions ##
###############

def rvs_spd_matrix( dim ):##{{{
	"""
	TODO move to tools
	"""
	O = sc.ortho_group.rvs(dim)
	S = np.diag(np.random.exponential(size = dim))
	return O @ S @ O.T
##}}}

def lmoments( X ):##{{{
	"""
	SBCK.tools.lmoments
	===================
	Compute the four first l-moments of X
	"""
	size = X.size
	
	C0 = scs.binom( range( size ) , 1 )
	C1 = scs.binom( range( size - 1 , -1 , -1 ) , 1 )
	
	## Order 3
	C2 = scs.binom( range( size ) , 2 )
	C3 = scs.binom( range( size - 1 , -1 , -1 ) , 2 )
	
	## Order 4
	C4 = scs.binom( range( size ) , 3 )
	C5 = scs.binom( range( size - 1 , -1 , -1 ) , 3 )
	
	M = np.zeros( (size,4) )
	M[:,0] = 1. / size
	M[:,1] = ( C0 - C1 ) / ( 2 * scs.binom( size , 2 ) )
	M[:,2] = ( C2 - 2 * C0 * C1 + C3 ) / ( 3 * scs.binom( size , 3 ) )
	M[:,3] = ( C4 - 3 * C2 * C1 + 3 * C0 * C3 - C5 ) / ( 4 * scs.binom( size , 4 ) )
	
	return (M.T @ np.sort(X).reshape(-1,1)).squeeze()
##}}}

def gpdfit( X , method = "lmoment" ):##{{{
	"""
	SBCK.tools.gpdfit
	=================
	Fit the scale and shape parameters of the GPD distribution of X, assuming
	the loc parameter is 0.
	"""
	if method == "lmoment":
		lmom  = lmoments(X)
		itau  = lmom[0] / lmom[1]
		scale = lmom[0] * ( itau - 1 )
		scale = scale if scale > 0 else 1e-8
		shape = 2 - itau
	else:
		m     = np.mean(X)
		s     = np.std(X)
		scale = m * ( m**2 / s**2 + 1 ) / 2
		shape = 1 - scale / m
	return scale,shape
##}}}


