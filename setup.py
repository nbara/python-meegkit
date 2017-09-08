#! /usr/bin/env python
#
from setuptools import setup, find_packages

setup(
    name='meegkit',
    description='M/EEG denoising in Python',
    url='http://github.com/eeginlscp/digitrack/',
    author='N Barascud',
    author_email='nicolas.barascud@ens.fr',
    license='UNLICENSED',
    version='0.1',
    packages=find_packages(exclude=['doc', 'tests']),
    zip_safe=False)
