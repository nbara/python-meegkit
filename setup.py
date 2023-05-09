#! /usr/bin/env python
#
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fid:
    long_description = fid.read()

setup(
    name='meegkit',
    description='M/EEG denoising in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/nbara/python-meegkit/',
    author='Nicolas Barascud',
    author_email='nicolas.barascud@gmail.com',
    license='BSD (3-clause)',
    version='0.1.3',
    packages=find_packages(exclude=['doc', 'tests']),
    project_urls={
        "Documentation": "https://nbara.github.io/python-meegkit/",
        "Source": "https://github.com/nbara/python-meegkit/",
        "Tracker": "https://github.com/nbara/python-meegkit/issues/",
    },
    platforms="any",
    python_requires=">=3.8",
    install_requires=["numpy", "scipy", "scikit-learn", "joblib", "pandas",
                      "matplotlib", "tqdm", "pyriemann", "statsmodels"],
    zip_safe=False)
