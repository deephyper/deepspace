#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

on_rtd = os.environ.get("READTHEDOCS") == "True"
on_theta = type(os.environ.get("HOST")) is str and "theta" in os.environ.get("HOST")
on_gpu = type(os.environ.get("DH_GPU")) is str and "true" == os.environ.get("DH_GPU")

# Package meta-data.
NAME = "deepspace"
DESCRIPTION = "A collection of search space for the DeepHyper package."
URL = "https://github.com/deephyper/deepspace"
REQUIRES_PYTHON = ">=3.7, <3.10"
VERSION = None

# Build Author list
authors = {
    "Prasanna Balaprakash": "pbalapra@anl.gov",
    "Romain Egele": "romain.egele@polytechnique.edu",
    "Romit Maulik": "rmaulik@anl.gov",
}
AUTHOR = ""
for i, (k, v) in enumerate(authors.items()):
    if i > 0:
        AUTHOR += ", "
    AUTHOR += f"{k} <{v}>"

# What packages are required for this module to be executed?
REQUIRED = ["deephyper"]

if on_rtd:
    REQUIRED.append("Sphinx>=1.8.2")
    REQUIRED.append("sphinx_rtd_theme")

# What packages are optional?
EXTRAS = {
    "tests": ["pytest", "codecov", "pytest-cov"],
    "dev": ["twine"],
    "docs": ["Sphinx>=1.8.2", "sphinx_rtd_theme"],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        sys.exit()


class TestUploadCommand(Command):
    """Support setup.py testupload."""

    description = "Build and publish the package to test.pypi."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload --repository-url https://test.pypi.org/legacy/ dist/*")

        sys.exit()


class TestInstallCommand(Command):
    """Support setup.py testinstall"""

    description = "Install deephyper from TestPyPI."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.status("Downloading the package from Test PyPI and installing it")
        os.system("pip install --index-url https://test.pypi.org/simple/ deephyper")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    project_urls={
        # "Documentation": "https://deephyper.readthedocs.io/",
        "Source": "https://github.com/deephyper/deepspace",
        "Tracker": "https://github.com/deephyper/deepspace/issues",
    },
    packages=find_packages(exclude=("tests",)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['deephyper'],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="ANL",
    classifiers=[
        # Trove classifiers
        # https://pypi.org/classifiers/
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
        "testupload": TestUploadCommand,
        "testinstall": TestInstallCommand,
    },
)
