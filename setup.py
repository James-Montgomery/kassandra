#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" setup script to install kassandra """

import setuptools
import versioneer

# Note: setup.cfg is set up to only recognise tags starting with v

# git describe --tags # gets the current tag
# git tag v0.02 # update the tag to something, e.g. v0.02
# git push origin --tags # push update to branch

# python3 -m pip install --user --upgrade setuptools wheel
# rm -rf build && rm -rf dist && rm -rf *.info
# python3 setup.py sdist bdist_wheel
# python3 -m twine upload --verbose --repository-url https://upload.pypi.org/legacy/ dist/*

# generate pylintrc file
# pylint --generate-rcfile > .pylintrc

################################################################################

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

with open("README.md", "r") as fh:
    """ load package description from readme file """
    long_description = fh.read()

################################################################################

def setup_package():
    """
    Function to manage setup procedures.
    """

    setuptools.setup(
        name="kassandra",
        version=versioneer.get_version(),

        author="James Montgomery",
        author_email="jamesoneillmontgomery@gmail.com",
        description="Package for implementing bayesian deep learning models in python.",
        long_description=long_description,
        long_description_content_type="text/markdown",

        #url="https://github.com/James-Montgomery/kassandra",
        packages=setuptools.find_packages(),
        install_requires=parse_requirements("requirements.txt"),
        extra_require={},
        cmdclass = versioneer.get_cmdclass()
    )

if __name__ == "__main__":
    setup_package()
