.. ``meegkit`` documentation master file, created by
   sphinx-quickstart on Fri Jan 10 12:31:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|:brain:| ``meegkit``: EEG and MEG denoising in Python
======================================================

Introduction
------------

``meegkit`` is a collection of EEG and MEG denoising techniques for
**Python 3.6+**. Please feel free to contribute, or suggest new analyses. Keep
in mind that this is mostly development code, and as such is likely to change
without any notice. Also, while most of the methods have been fairly robustly
tested, bugs can (and should!) be expected.

The source code of the project is hosted on Github at the following address:
https://github.com/nbara/python-meegkit

To get started, follow the installation instructions `in the README <https://github.com/nbara/python-meegkit#installation>`_.

Available modules
-----------------

Here is a list of the methods and techniques available in ``meegkit``:

.. currentmodule:: meegkit

.. toctree::
   :maxdepth: 1

.. autosummary::
   :caption: meegkit

   ~meegkit.asr
   ~meegkit.cca
   ~meegkit.dss
   ~meegkit.detrend
   ~meegkit.ress
   ~meegkit.sns
   ~meegkit.star
   ~meegkit.trca
   ~meegkit.tspca
   ~meegkit.utils


Examples gallery
----------------

A number of example scripts and notebooks are available:


.. toctree::
   :maxdepth: 2

   auto_examples/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
