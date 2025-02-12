#!/usr/bin/env python3 -m unittest

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


#############
## Imports ##
#############

import os
import itertools as itt
import unittest

import numpy as np
import scipy.stats as sc
import xarray as xr
import pandas as pd

import SBCK as bc
import SBCK.ppp as bcp
import SBCK.tools as bct
import SBCK.datasets as bcd
import SBCK.metrics as bcm

import matplotlib as mpl
import matplotlib.pyplot   as plt
import matplotlib.gridspec as mplg


########################
## Set mpl parameters ##
########################

mpl.rcdefaults()
mpl.rcParams['font.size'] = 7
mpl.rcParams['axes.linewidth']  = 0.5
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['patch.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5


######################
## Parameters class ##
######################

class SBCKTestParameters:##{{{
	
	def __init__( self ):##{{{
		lpath = os.path.join( *os.path.basename(__file__).split(".")[0].split("_")[1:] )
		self.opath = os.path.join( os.path.dirname(__file__) , "figures" , lpath )
		if not os.path.isdir(self.opath):
			os.makedirs(self.opath)
	##}}}
	
	@property
	def prefix(self):
		return str(type(self)).split("'")[1].split(".")[-1]
	
##}}}


###############
## PPP Tests ##
###############

class Test_PrePostProcessing(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_identity_CDFt(self):##{{{
		
		## Parameters
		size = 10_000
		
		## Data
		Y0,X0,X1 = bcd.like_tas_pr(size)
		
		## Parameters of the ppp
		bc_method        = bc.CDFt
		bc_method_kwargs = {}
		pipe             = []
		pipe_kwargs      = []
		
		## Correction with the ppp, but ppp is just the identity
		cppp  = bcp.PrePostProcessing( bc_method = bc_method , bc_method_kwargs = bc_method_kwargs , pipe = pipe , pipe_kwargs = pipe_kwargs ).fit( Y0 , X0 , X1 )
		Z1p,Z0p = cppp.predict( X1 , X0 )
		
		## Direct correction
		corr = bc_method( **bc_method_kwargs ).fit( Y0 , X0 , X1 )
		Z1d,Z0d = corr.predict( X1 , X0 )
		
		## Comparison
		D1 = np.abs(Z1p - Z1d).max()
		D0 = np.abs(Z0p - Z0d).max()
		
		self.assertAlmostEqual( D1 , 0 )
		self.assertAlmostEqual( D0 , 0 )
	##}}}
	
##}}}

class Test_SSR(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_pr_CDFt(self):##{{{
		
		## Parameters
		size = 10_000
		
		## Data
		np.random.seed(42)
		bY0 = 1
		bX0 = 0.5
		bX1 = 0.25
		Y0  = np.random.gamma( shape = 0.5 , scale = 2 , size = size ) + bY0
		X0  = np.random.exponential( scale = 1 , size = size )         + bX0
		X1  = np.random.exponential( scale = 0.5 , size = size )       + bX1
		
		## Add zero
		Y0[np.random.choice( size , int(0.4*size) , replace = False )] = 0
		X0[np.random.choice( size , int(0.05*size) , replace = False )] = 0
		X1[np.random.choice( size , int(0.15*size) , replace = False )] = 0
		
		## Parameters of the ppp
		bc_method        = bc.CDFt
		bc_method_kwargs = { "rvY" : bct.rv_empirical , "rvX" : bct.rv_empirical , "norm" : "d-quant" , "oob" : "None" }
		pipe             = [bcp.SSR]
		pipe_kwargs      = [{'cols' : 0}]
		
		## Correction with the ppp
		cppp  = bcp.PrePostProcessing( bc_method = bc_method , bc_method_kwargs = bc_method_kwargs , pipe = pipe , pipe_kwargs = pipe_kwargs ).fit( Y0 , X0 , X1 )
		Z1p,Z0p = cppp.predict( X1 , X0 )
		
		## Direct correction
		Z1d,Z0d = bc_method( **bc_method_kwargs ).fit( Y0 , X0 , X1 ).predict( X1 , X0 )
		
#		Z1d[Z1d < bY0] = 0
		
		## Random variable
		rvY0  = bct.rv_empirical( X = Y0  )
		rvX0  = bct.rv_empirical( X = X0  )
		rvX1  = bct.rv_empirical( X = X1  )
		rvZ1p = bct.rv_empirical( X = Z1p )
		rvZ0p = bct.rv_empirical( X = Z0p )
		rvZ1d = bct.rv_empirical( X = Z1d )
		rvZ0d = bct.rv_empirical( X = Z0d )
		rvZ0  = [rvZ0p,rvZ0d]
		rvZ1  = [rvZ1p,rvZ1d]
		
		xmin    = min([rv.a for rv in rvZ0 + rvZ1 + [rvY0,rvX0,rvX1]])
		xmax    = max([rv.b for rv in rvZ0 + rvZ1 + [rvY0,rvX0,rvX1]])
		x       = np.linspace( xmin , xmax , 1000 )
		link    = lambda x: np.log( 1 + np.array([x]).squeeze() )
		l1x     = link(x)
		xticks  = np.sort( np.arange(0,11,2.5).tolist() + [bY0] ) 
		lxticks = link(xticks)
		xlim    = link([xmin,11])
		
		##
		fig = plt.figure( dpi = 120 )
		grid = mplg.GridSpec( 3 , 3 )
		
		ax = fig.add_subplot(grid[1,1])
		ax.plot( l1x ,  rvY0.cdf(x) , label = r"$Y^0$" , color = "blue"   )
		ax.plot( l1x ,  rvX0.cdf(x) , label = r"$X^0$" , color = "red"    )
		ax.plot( l1x ,  rvX1.cdf(x) , label = r"$X^1$" , color = "purple" )
		ax.plot( l1x , rvZ1p.cdf(x) , label = r"$Z^1$ (PPP)" , color = "darkgreen" , linestyle = "-" )
		ax.plot( l1x , rvZ1d.cdf(x) , label = r"$Z^1$ (DIR)" , color = "darkgreen" , linestyle = "--" )
		ax.axvline( link(bY0) , color = "blue"   , linestyle = ":" )
		ax.axvline( link(bX0) , color = "red"    , linestyle = ":" )
		ax.axvline( link(bX1) , color = "purple" , linestyle = ":" )
		ax.set_xticks(lxticks)
		ax.set_xticklabels(xticks)
		ax.set_xlim(xlim)
		ax.legend( loc = "lower right" )
		
		## Figsize
		mm = 1. / 25.4
		pt = 1. / 72
		
		width  = 210*mm
		w_l    = 30*pt
		w_m    = 35*pt
		w_r    =  5*pt
		w_ax   = (width - (w_l + w_r)) / 1
		widths = [w_l,w_ax,w_r]
		
		h_ax    = w_ax / (16 / 11)
		h_t     = 20*pt
		h_m     = 30*pt
		h_b     = 30*pt
		heights = [h_t,h_ax,h_b]
		height  = sum(heights)
		
		grid.set_height_ratios(heights)
		grid.set_width_ratios(widths)
		fig.set_figheight(height)
		fig.set_figwidth(width)
		
		plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
		plt.savefig( os.path.join( self.opath , f"{self.prefix}_rv_method.png" ) , dpi = 600 )
		plt.close(fig)
		
	##}}}
	
##}}}

class Test_OTCNoise(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_add_noise(self):##{{{
		
		## Parameters
		size = 5_000
		
		## Data
		np.random.seed(42)
		Y0,X0,X1 = bc.datasets.gaussian_L_2d(size)
		
		## Parameters of the ppp
		bc_method        = bc.dOTC
		bc_method_kwargs = { "cov_factor" : "cholesky" }
		pipe             = [bcp.OTCNoise]
		pipe_kwargs      = [{}]
		
		## Correction with the ppp
		cppp  = bcp.PrePostProcessing( bc_method = bc_method , bc_method_kwargs = bc_method_kwargs , pipe = pipe , pipe_kwargs = pipe_kwargs ).fit( Y0 , X0 , X1 )
		Z1p,Z0p = cppp.predict( X1 , X0 )
		
		## Direct correction
		Z1d,Z0d = bc_method( **bc_method_kwargs ).fit( Y0 , X0 , X1 ).predict( X1 , X0 )
		
		## xylim
		xylim = [-5,11]
		dxy   = 0.1
		bins  = np.linspace( xylim[0] , xylim[1] , 100 )
		
		## Figure
		titles = [r"$\mathbf{Y}^0$",r"$\mathbf{X}^0$",r"$\mathbf{Z}^1$",r"$\mathbf{X}^1$"]
		colors = ["blue","red","purple","green"]
		fig = plt.figure()
		grid = mplg.GridSpec(5,5)
		for ij,K in enumerate([Z0p,Z0d,Z1p,Z1d]):
			j = ij  % 2
			i = ij // 2
			ax  = fig.add_subplot(grid[2*i+1,2*j+1])
			for iK,K in enumerate([Y0,X0,X1,K]):
				ax.plot( K[:,0] , K[:,1] , color = colors[iK] , linestyle = "" , marker = "." , markersize = 0.5 )
#			ax.text( xylim[1] - 3 * dxy , xylim[1] - 3 * dxy , titles[iK] , ha = "right" , va = "top" , fontdict = { "size" : 12 } , bbox = { 'facecolor' : 'none' , 'edgecolor' : 'black' , "boxstyle" : 'round' } )
			
			if i == 1:
				ax.set_xlabel(r"$x_0$")
			else:
				ax.set_xticks([])
			if j == 0:
				ax.set_ylabel(r"$x_1$")
			else:
				ax.set_yticks([])
			ax.spines[['right', 'top']].set_visible(False)
			ax.set_xlim(xylim)
			ax.set_ylim(xylim)
		
		## Title
		ax = fig.add_subplot(grid[0,1:4])
		ax.text( 0 , 0 , "dOTC noise" , ha = "center" , va = "center" , fontdict = { "weight" : "bold" , "size" : 12 } )
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.set_axis_off()
		
		## Figsize
		mm = 1. / 25.4
		pt = 1. / 72
		
		width  = 180*mm
		w_l    = 30*pt
		w_m    =  5*pt
		w_r    =  5*pt
		w_ax   = (width - (w_l + w_m + w_r)) / 2
		widths = [w_l,w_ax,w_m,w_ax,w_r]
		
		h_ax    = w_ax
		h_t     = 17*pt
		h_m     =  5*pt
		h_b     = 30*pt
		heights = [h_t,h_ax,h_m,h_ax,h_b]
		height  = sum(heights)
		
		grid.set_height_ratios(heights)
		grid.set_width_ratios(widths)
		fig.set_figheight(height)
		fig.set_figwidth(width)
		plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
		
		plt.savefig( os.path.join( self.opath , f"{self.prefix}_add_noise.png" ) , dpi = 600 )
		plt.close(fig)
		
	##}}}
	
##}}}

class Test_LinkFunction(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_plot_LF(self):##{{{
		
		##
		xlim   = (-5,5)
		ylim   = (-5,5)
		x      = np.linspace( xlim[0] , xlim[1] , 10_000 )
		kwargs = [ { "m" : 3 } , {"s" : 0.5} , {"M" : 3} , {"M" : 2} , {"s" : 0.5} , {"ymin" : -1 , "ymax" : 3} , {"ymin" : -1 , "ymax" : 3 , "s" : 10 } ]
		
		## Loop on link function
		for ilf,LF in enumerate([bcp.LFAdd,bcp.LFMult,bcp.LFMax,bcp.LFMin,bcp.LFLoglin,bcp.LFArctan,bcp.LFLogistic]):
			
			fig = plt.figure()
			grid = mplg.GridSpec( 3 , 5 )
			
			lf = LF(**kwargs[ilf])
			
			ax = fig.add_subplot(grid[1,1])
			ax.plot( x , lf.transform(x) , color = "blue" )
			ax.plot( xlim , ylim , color = "black" , linestyle = ":" )
			ax.set_xlabel( r"$x$" )
			ax.set_ylabel( r"$y$" )
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
			
			ax = fig.add_subplot(grid[1,3])
			ax.plot( x , lf.itransform(x) , color = "red" )
			ax.plot( ylim , xlim , color = "black" , linestyle = ":" )
			ax.set_xlabel( r"$y$" )
			ax.set_ylabel( r"$x$" )
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
			
			## Title
			ax = fig.add_subplot(grid[0,1:-1])
			ax.text( 0 , 0 , lf.name , ha = "center" , va = "center" , fontdict = { "weight" : "bold" , "size" : 12 } )
			ax.set_xlim(-1,1)
			ax.set_ylim(-1,1)
			ax.set_axis_off()
			
			## Figsize
			mm = 1. / 25.4
			pt = 1. / 72
			
			width  = 180*mm
			w_l    = 30*pt
			w_m    = 30*pt
			w_r    =  5*pt
			w_ax   = (width - (w_l + w_m + w_r)) / 2
			widths = [w_l,w_ax,w_m,w_ax,w_r]
			
			h_ax    = w_ax / (4/3)
			h_t     = 17*pt
			h_b     = 30*pt
			heights = [h_t,h_ax,h_b]
			height  = sum(heights)
			
			grid.set_height_ratios(heights)
			grid.set_width_ratios(widths)
			fig.set_figheight(height)
			fig.set_figwidth(width)
			plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
			
			plt.savefig( os.path.join( self.opath , f"{self.prefix}_{lf.name}_plot_LF.png" ) , dpi = 600 )
			plt.close(fig)
		
	##}}}
	
##}}}

class Test_XarrayAs2d(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_keep_struct(self):##{{{
		
		##
		time     = xr.date_range( "2000-01-01","2005-12-31" ).values
		cvars    = ["tas","pr"]
		size     = time.size
		Y0,X0,X1 = bc.datasets.like_tas_pr(size)
		
		## Add xarray structure
		Y0 = xr.DataArray( Y0 , dims = ["time","cvar"] , coords = [time,cvars] , attrs = { "kind" : "Y0" } )
		X0 = xr.DataArray( X0 , dims = ["time","cvar"] , coords = [time,cvars] , attrs = { "kind" : "X0" } )
		X1 = xr.DataArray( X1 , dims = ["time","cvar"] , coords = [time,cvars] , attrs = { "kind" : "X1" } )
		
		## Parameters of the ppp
		bc_method        = bc.IdBC
		bc_method_kwargs = {}
		pipe             = [bcp.Xarray]
		pipe_kwargs      = [{}]
		
		## Correction with the ppp
		cppp  = bcp.PrePostProcessing( bc_method = bc_method , bc_method_kwargs = bc_method_kwargs , pipe = pipe , pipe_kwargs = pipe_kwargs ).fit( Y0 , X0 , X1 )
		Z1,Z0 = cppp.predict( X1 , X0 )
		
		self.assertLess( np.abs(Z0 - X0).max() , 1e-6 )
		self.assertLess( np.abs(Z1 - X1).max() , 1e-6 )
		
	##}}}
	
	def test_keep_struct_2d(self):##{{{
		
		##
		time     = xr.date_range( "2000-01-01","2005-12-31" ).values
		lat      = np.array([30,60])
		lon      = np.array([-90,90])
		cvars    = ["tas","pr"]
		size     = time.size
		Y0,X0,X1 = bc.datasets.like_tas_pr(lat.size * lon.size * size)
		
		## Add xarray structure
		Y0 = xr.DataArray( Y0.reshape(time.size,lat.size,lon.size,2) , dims = ["time","lat","lon","cvar"] , coords = [time,lat,lon,cvars] , attrs = { "kind" : "Y0" } )
		X0 = xr.DataArray( X0.reshape(time.size,lat.size,lon.size,2) , dims = ["time","lat","lon","cvar"] , coords = [time,lat,lon,cvars] , attrs = { "kind" : "X0" } )
		X1 = xr.DataArray( X1.reshape(time.size,lat.size,lon.size,2) , dims = ["time","lat","lon","cvar"] , coords = [time,lat,lon,cvars] , attrs = { "kind" : "X1" } )
		
		## Parameters of the ppp
		bc_method        = bc.IdBC
		bc_method_kwargs = {}
		pipe             = [bcp.As2d,bcp.Xarray]
		pipe_kwargs      = [{},{}]
		
		## Correction with the ppp
		cppp  = bcp.PrePostProcessing( bc_method = bc_method , bc_method_kwargs = bc_method_kwargs , pipe = pipe , pipe_kwargs = pipe_kwargs ).fit( Y0 , X0 , X1 )
		Z1,Z0 = cppp.predict( X1 , X0 )
		
		self.assertLess( np.abs(Z0 - X0).max() , 1e-6 )
		self.assertLess( np.abs(Z1 - X1).max() , 1e-6 )
		
	##}}}
##}}}

class Test_OnlyFinite(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_onlyFinite(self):##{{{
		
		##
		size     = 1000
		Y0,X0,X1 = bc.datasets.like_tas_pr(size)
		
		p0   = 0.1
		p1   = 0.04
		idx0 = np.random.choice( size , int(size * p0) , replace = False )
		idx1 = np.random.choice( size , int(size * p1) , replace = False )
		
		X0[idx0,1] = np.nan
		X1[idx1,1] = np.nan
		
		## Parameters of the ppp
		bc_method        = bc.CDFt
		bc_method_kwargs = {}
		pipe             = [bcp.OnlyFinite]
		pipe_kwargs      = [{}]
		
		## Raise exception
		cppp = bcp.PrePostProcessing( bc_method = bc_method , bc_method_kwargs = bc_method_kwargs , pipe = pipe , pipe_kwargs = pipe_kwargs )
		self.assertRaises( ValueError , cppp.fit , *(Y0,X0,X1) )
		
		## Good check function
		cppp = bcp.PrePostProcessing( bc_method = bc_method , bc_method_kwargs = bc_method_kwargs , pipe = pipe , pipe_kwargs = pipe_kwargs , checkf = bcp.atleastonefinite ).fit( Y0 , X0 , X1 )
		Z1,Z0 = cppp.predict( X1 , X0 )
		
		ep0 = float((1 - np.isfinite(Z0[:,1]).sum() / Z0[:,1].shape)[0])
		ep1 = float((1 - np.isfinite(Z1[:,1]).sum() / Z1[:,1].shape)[0])
		self.assertAlmostEqual( ep0 , p0 )
		self.assertAlmostEqual( ep1 , p1 )
		
	##}}}
	
	def test_onlyFiniteAnalog(self):##{{{
		
		##
		size     = 1000
		Y0,X0,X1 = bc.datasets.like_tas_pr(size)
		
		p0   = 0.1
		p1   = 0.04
		idx0 = np.random.choice( size , int(size * p0) , replace = False )
		idx1 = np.random.choice( size , int(size * p1) , replace = False )
		
		X0[idx0,1] = np.nan
		X1[idx1,1] = np.nan
		
		## Parameters of the ppp
		bc_method        = bc.CDFt
		bc_method_kwargs = {}
		pipe             = [bcp.OnlyFiniteAnalog]
		pipe_kwargs      = [{ "analog_var" : 0 }]
		
		## Good check function
		cppp = bcp.PrePostProcessing( bc_method = bc_method , bc_method_kwargs = bc_method_kwargs , pipe = pipe , pipe_kwargs = pipe_kwargs , checkf = bcp.atleastonefinite ).fit( Y0 , X0 , X1 )
		Z1,Z0 = cppp.predict( X1 , X0 )
		
		ep0 = float((1 - np.isfinite(Z0[:,1]).sum() / Z0[:,1].shape)[0])
		ep1 = float((1 - np.isfinite(Z1[:,1]).sum() / Z1[:,1].shape)[0])
		self.assertAlmostEqual( ep0 , p0 )
		self.assertAlmostEqual( ep1 , 0  )
		
	##}}}
	
##}}}

class Test_Extremes(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def find_ratio( self , K1 , K0 , p = 0.95 ):##{{{
		uK0 = np.quantile( K0 , p )
		uK1 = np.quantile( K1 , p )
		Rl  = K1[K1 < uK1].mean() / K0[K0 < uK0].mean()
		Ru  = K1[K1 > uK1].mean() / K0[K0 > uK0].mean()
		
		return uK1,uK0,Rl,Ru
	##}}}
	
	def test_LimitTaislRatio(self):##{{{
		
		##
		np.random.seed(42)
		
		## Data
		size = 10_000
		Y0   = np.random.exponential( scale = 1   , size = size )
		X0   = np.random.exponential( scale = 0.1 , size = size )
		X1   = np.random.exponential( scale = 3   , size = size )
		p    = 0.99
		
		## Transform the right tail
		uX1,uX0,RXl,RXr = self.find_ratio( X1 , X0 , p )
		X1[X1 > uX1]    = X1[X1 > uX1] * 10 * RXl
		
		## Parameters of the ppp
		bc_method    = bc.CDFt
		bcmkws       = [{ "norm" : "origin"  , "oob" : "Y0CC" },{ "norm" : "d-quant" , "oob" : "None" , "norm_e" : 1-p }]
		pipes        = [ [], [ bcp.LimitTailsRatio , bcp.As2d ]]
		pipes_kwargs = [ [], [ { "tails" : "right" , "cols" : 0 } , {} ]]
		
		## Output
		df = pd.DataFrame( np.nan , columns = pd.MultiIndex.from_product([range(2),range(2)]) , index = ["RXl","RXr","RZl","RZr","Y0x","X0x","X1x","Z0x","Z1x","Y0q","X0q","X1q","Z0q","Z1q"] )
		df.loc["RXl",:] = RXl
		df.loc["RXr",:] = RXr
		df.loc["Y0x",:] = Y0.max()
		df.loc["X0x",:] = X0.max()
		df.loc["X1x",:] = X1.max()
		df.loc["Y0q",:] = np.quantile( Y0 , p )
		df.loc["X0q",:] = np.quantile( X0 , p )
		df.loc["X1q",:] = np.quantile( X1 , p )
		
		##
		rvY0 = bct.rv_empirical( X = Y0 )
		rvX0 = bct.rv_empirical( X = X0 )
		rvX1 = bct.rv_empirical( X = X1 )
		x    = np.linspace( -5 , 20 , 1000 )
		names_norm = ["norm_origin","norm_d-quant"]
		names_ppp  = ["no-ppp","LimitTailsRatio"]
		
		## Loop
		fig = plt.figure()
		grid = mplg.GridSpec( 2 * 2 + 1 , 2 * 2 + 1 )
		for i,j in itt.product(range(2),range(2)):
			
			## Correction
			bcmkw       = bcmkws[i]
			pipe        = pipes[j]
			pipe_kwargs = pipes_kwargs[j]
			
			cppp = bcp.PrePostProcessing( bc_method = bc_method , bc_method_kwargs = bcmkw , pipe = pipe , pipe_kwargs = pipe_kwargs , checkf = bcp.atleastonefinite ).fit( Y0 , X0 , X1 )
			Z1,Z0 = cppp.predict( X1 , X0 )
			
			## Add to table
			_,_,Rl,Rr = self.find_ratio( Z1 , Z0 , p )
			df.loc["RZl",(i,j)] = Rl
			df.loc["RZr",(i,j)] = Rr
			df.loc["Z0x",(i,j)] = Z0.max()
			df.loc["Z1x",(i,j)] = Z1.max()
			df.loc["Z0q",(i,j)] = np.quantile( Z0 , p )
			df.loc["Z1q",(i,j)] = np.quantile( Z1 , p )
			
			##
			rvZ1 = bct.rv_empirical( X = Z1 )
			
			## Add to figure
			ax = fig.add_subplot( grid[2*i+1,2*j+1] )
			ax.plot( x , rvY0.cdf(x) , color = "blue"   )
			ax.plot( x , rvX0.cdf(x) , color = "red"    )
			ax.plot( x , rvX1.cdf(x) , color = "purple" )
			ax.plot( x , rvZ1.cdf(x) , color = "green"  , marker = "x" )
			ax.set_title( f"{names_norm[i]} / {names_ppp[j]}" )
			ax.set_ylim(0,1)
			
			if j == 0:
				ax.set_ylabel( "CDF" )
			else:
				ax.set_yticks([])
			if i == 1:
				ax.set_xlabel( r"$x$" )
			else:
				ax.set_xticks([])
			ax.spines[['right', 'top']].set_visible(False)
		
		## Tables output
		df.columns = pd.MultiIndex.from_product( [names_norm,names_ppp] )
		
		with open( os.path.join( self.opath , f"{self.prefix}_LimitTailsRatio.txt" ) , "w" ) as f:
			f.write( df.to_string() )
		
		## Figsize
		mm = 1. / 25.4
		pt = 1. / 72
		
		width  = 180*mm
		w_l    = 30*pt
		w_m    =  5*pt
		w_r    =  5*pt
		w_ax   = (width - (w_l + w_m + w_r)) / 2
		widths = [w_l,w_ax,w_m,w_ax,w_r]
		
		h_ax    = w_ax / (4/3)
		h_t     = 17*pt
		h_m     = 17*pt
		h_b     = 30*pt
		heights = [h_t,h_ax,h_m,h_ax,h_b]
		height  = sum(heights)
		
		grid.set_height_ratios(heights)
		grid.set_width_ratios(widths)
		fig.set_figheight(height)
		fig.set_figwidth(width)
		plt.subplots_adjust( left = 0 , right = 1 , top = 1 , bottom = 0 , wspace = 0 , hspace = 0 )
		plt.savefig( os.path.join( self.opath , f"{self.prefix}_LimitTailsRatio.png" ) , dpi = 600 )
		plt.close(fig)
		
	##}}}
	
##}}}

class Test_DeltaRef(SBCKTestParameters,unittest.TestCase):##{{{
	
	def __init__( self , *args, **kwargs ):##{{{
		SBCKTestParameters.__init__( self )
		unittest.TestCase.__init__( self , *args , **kwargs )
	##}}}
	
	def test_preserveOrder(self):##{{{
		
		##
		np.random.seed(42)
		
		ppp = bcp.PreserveOrder( cols = [1,2,0] )
		
		X  = np.random.normal( size = (1000,3) )
		Xt = ppp.itransform(ppp.transform(X))
		
		self.assertTrue( ( (Xt[:,1] < Xt[:,2]) & (Xt[:,2] < Xt[:,0]) ).all() )
		
	##}}}
	
##}}}


##########
## main ##
##########

if __name__ == "__main__":
	unittest.main()
