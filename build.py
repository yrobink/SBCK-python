import os
import shutil
import site
import sys
import sysconfig
import tomllib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pybind11
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

cpath = Path(__file__).parent

# Load custom Eigen settings
with open("pyproject.toml", "rb") as f:
    toml = tomllib.load(f)
toml_eigen = toml["eigen"]
eigen_usr_include, eigen_auto_install = toml_eigen.get("usr_include", ""), toml_eigen.get("auto_install", False)


def install_eigen_locally():
    """install eigen-3.4.0 headers locally"""
    url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    download_dir = "./downloads"
    extraction_dir = "./extraction"
    destination_dir = os.path.join(site.getsitepackages()[0], "include")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extraction_dir, exist_ok=True)
    zip_file_path = os.path.join(download_dir, "eigen-3.4.0.zip")
    urlretrieve(url, zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)
    source_dir = os.path.join(extraction_dir, "eigen-3.4.0/Eigen")
    os.makedirs(destination_dir, exist_ok=True)
    shutil.move(source_dir, destination_dir)
    shutil.rmtree(download_dir)
    shutil.rmtree(extraction_dir)


if eigen_auto_install: install_eigen_locally()


def get_eigen_include(propose_path=""):
    """Attempt to find Eigen headers include directory"""
    possible_path = [propose_path, os.path.dirname(sysconfig.get_paths()['include']), "/usr/include/", "/usr/local/include/"]
    if os.environ.get("HOME") is not None:
        possible_path.append(os.path.join(os.environ["HOME"], ".local/include"))

    for path in possible_path:
        eigen_include = os.path.join(path, "Eigen")
        if os.path.isdir(eigen_include):
            return path

        eigen_include = os.path.join(path, "eigen3", "Eigen")
        if os.path.isdir(eigen_include):
            return os.path.join(path, "eigen3")

    return ""


def has_flag(compiler, flagname):
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


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        opts.append("-O3")
        if ct == 'unix':
            opts.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


# Extension to compile
ext_modules = [
    Extension(
        "SBCK.tools.__tools_cpp",
        [str(cpath / 'SBCK/tools/src/tools.cpp')],
        include_dirs=[  # Path to pybind11 headers
            get_eigen_include(eigen_usr_include),
            pybind11.get_include(),
            pybind11.get_include(user=True),
        ],
        language='c++',
        depends=["SBCK/tools/src/SparseHist.hpp"]
    ),
]

# Update the description for pypi
long_description = (cpath / "README.md").read_text()
for i, f in [(":heavy_check_mark:", " Yes              "), (":x: ", " No "), (":warning:", " ~       ")]:
    long_description = long_description.replace(i, f)


# update generated setup.py args
def build(setup_args: dict):
    setup_args.update(dict(
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExt},
        zip_safe=False,
        package_dir={"SBCK": "./SBCK"},
        long_description_content_type='text/markdown',
        long_description=long_description,
    ))
