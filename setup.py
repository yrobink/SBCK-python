# -*- coding: utf-8 -*-

## Copyright(c) 2021 / 2023 Yoann Robin
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

import os
import sys
import sysconfig
import subprocess
import tempfile
import uuid
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import setuptools
from pathlib import Path


#####################
## User Eigen path ##
#####################

eigen_usr_include = os.environ.get('EIGEN_PATH', '')

i_eigen = -1
for i,arg in enumerate(sys.argv):
	if arg[:5] == "eigen":
		eigen_usr_include = arg[6:]
		i_eigen = i

if i_eigen > -1:
	del sys.argv[i_eigen]
	
############################
## Python path resolution ##
############################

cpath = Path(__file__).parent


################################################################
## Some class and function to compile with Eigen and pybind11 ##
################################################################

class get_pybind_include(object):##{{{
	"""Helper class to determine the pybind11 include path
	The purpose of this class is to postpone importing pybind11
	until it is actually installed, so that the ``get_include()``
	method can be invoked. """
	
	def __init__(self, user=False):
		self.user = user
	
	def __str__(self):
		import pybind11
		return pybind11.get_include(self.user)
##}}}

def get_eigen_include( propose_path = "" ):##{{{
	possible_path = [propose_path, os.path.dirname(sysconfig.get_paths()['include']), "/usr/include/",
					 "/usr/local/include/", "external/eigen"]
	if os.environ.get("HOME") is not None:
		possible_path.append( os.path.join( os.environ["HOME"] , ".local/include" ) )
	
	for path in possible_path:
		
		
		eigen_include = os.path.join( path , "Eigen" )
		if os.path.isdir( eigen_include ):
			return path
		
		eigen_include = os.path.join( path , "eigen3" , "Eigen" )
		if os.path.isdir( eigen_include ):
			return os.path.join( path , "eigen3" )
	
	return ""
##}}}

def has_flag(compiler, flagname):##{{{
	"""Return a boolean indicating whether a flag name is supported on
	the specified compiler.
	"""
	import tempfile
	with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
		f.write('int main (int argc, char **argv) { return 0; }')
		try:
			compiler.compile([f.name], extra_postargs=[flagname])
		except setuptools.distutils.errors.CompileError:
			return False
	return True
##}}}

def cpp_flag(compiler):##{{{
	"""Return the -std=c++[11/14] compiler flag.
	The c++14 is prefered over c++11 (when it is available).
	"""
	if has_flag(compiler, '-std=c++14'):
		return '-std=c++14'
	elif has_flag(compiler, '-std=c++11'):
		return '-std=c++11'
	else:
		raise RuntimeError( 'Unsupported compiler -- at least C++11 support is needed!' )
##}}}

class BuildExt(build_ext):##{{{
	"""A custom build extension for adding compiler-specific options."""
	c_opts = {
		'msvc': ['/EHsc'],
		'unix': [],
	}
	
	if sys.platform == 'darwin':
		c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

	def initialize_options(self):
		temp_dir = tempfile.gettempdir()
		random_suffix = str(uuid.uuid4())  # More robust and unique than using pid or uid
		self.build_temp = os.path.join(temp_dir, f'sbck_build_temp_{random_suffix}')
		super().initialize_options()

	def run(self):
		# Ensure the Eigen submodule is initialized and updated
		if not os.path.exists("external/eigen"):
			print("Updating Eigen submodule...")
			subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], check=True)

		# Call the original run method to proceed with the build
		super().run()
	
	def build_extensions(self):
		ct = self.compiler.compiler_type
		opts = self.c_opts.get(ct, [])
		if ct == 'unix':
			opts.append( "-O3" )
			opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
			opts.append(cpp_flag(self.compiler))
			if has_flag(self.compiler, '-fvisibility=hidden'):
				opts.append('-fvisibility=hidden')
		elif ct == 'msvc':
			opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
		for ext in self.extensions:
			ext.extra_compile_args = opts
		build_ext.build_extensions(self)
##}}}


##########################
## Extension to compile ##
##########################

ext_modules = [
	Extension(
		"SBCK.tools.__tools_cpp",
		[ str(cpath / 'SBCK/tools/src/tools.cpp') ],
		include_dirs=[
			# Path to pybind11 headers
			get_eigen_include(eigen_usr_include),
			get_pybind_include(),
			get_pybind_include(user=True)
		],
		language='c++',
		depends = [
			"SBCK/tools/src/SparseHist.hpp"
			]
	),
]


#################
## Compilation ##
#################

list_packages = [
	"SBCK",
	"SBCK.ppp",
	"SBCK.tools",
	"SBCK.metrics",
	"SBCK.datasets"
]

########################
## Infos from release ##
########################

exec( (cpath / "SBCK" / "__release.py").read_text() )


#################
## Description ##
#################
long_description = (cpath / "README.md").read_text()
long_description = long_description.replace(":heavy_check_mark:"," Yes              ")
long_description = long_description.replace(":x: "," No ")
long_description = long_description.replace(":warning:"," ~       ")


#######################
## And now the setup ##
#######################

setup(
	name         = name,
	description  = description,
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	version      = version,
	author       = author,
	author_email = author_email,
	license      = license,
	platforms        = [ "linux" , "macosx" ],
	classifiers      = [
		"Development Status :: 5 - Production/Stable",
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		"Natural Language :: English",
		"Operating System :: MacOS :: MacOS X",
		"Operating System :: POSIX :: Linux",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Topic :: Scientific/Engineering :: Mathematics"
	],
	ext_modules      = ext_modules,
	install_requires = ["numpy" , "scipy" , "matplotlib" , "pybind11>=2.2" , "pot>=0.9.0"],
	cmdclass         = {'build_ext': BuildExt},
	zip_safe         = False,
	packages         = list_packages,
	package_dir      = { "SBCK" : "./SBCK" }
)


