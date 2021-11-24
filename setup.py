#! /usr/bin/env python
#
from setuptools import setup, find_packages

setup(
    name='meegkit',
    description='M/EEG denoising in Python',
    url='http://github.com/nbara/python-meegkit/',
    author='N Barascud',
    author_email='nicolas.barascud@gmail.com',
    license='UNLICENSED',
    version='0.1.2',
    packages=find_packages(exclude=['doc', 'tests']),
    zip_safe=False)
