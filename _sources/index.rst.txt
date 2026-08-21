.. ``meegkit`` documentation master file, created by
   sphinx-quickstart on Fri Jan 10 12:31:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|:brain:| ``meegkit``: EEG and MEG denoising in Python
======================================================

Introduction
------------

``meegkit`` is a collection of EEG and MEG denoising techniques for
**Python 3.8+**. Please feel free to contribute, or suggest new analyses. Keep
in mind that this is mostly development code, and as such is likely to change
without any notice. Also, while most of the methods have been fairly robustly
tested, bugs can (and should!) be expected.

The package is most useful for readers who want practical reference
implementations of denoising and component-analysis methods, together with
worked examples that show how to interpret the outputs.

The source code of the project is hosted on Github at the following address:
https://github.com/nbara/python-meegkit

Quick start
-----------

Install the package with ``pip``:

.. code-block:: bash

   pip install meegkit

Some ASR-related functionality requires optional dependencies. To install those
as well, use:

.. code-block:: bash

   pip install 'meegkit[extra]'

For development, documentation building, or testing, see the fuller
installation guidance `in the README <https://github.com/nbara/python-meegkit#installation>`_.

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
   ~meegkit.lof
   ~meegkit.phase
   ~meegkit.ress
   ~meegkit.sns
   ~meegkit.star
   ~meegkit.trca
   ~meegkit.tspca
   ~meegkit.utils

Examples gallery
----------------

A number of example scripts and notebooks are available.

If you are new to the package, a good starting sequence is:

1. ``example_asr`` for a full artifact-removal workflow.
2. ``example_dss`` for a simple synthetic component-recovery example.
3. ``example_trca`` or ``example_ress`` for task-oriented spatial filtering.

Many examples are synthetic sanity checks with known ground truth, which makes
them useful for understanding what each method is expected to recover.


.. toctree::
   :maxdepth: 2

   auto_examples/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
