AI/ML Workflows on the National CI
==================================

Companion guide for the PEARC26 tutorial *"AI/ML Workflows on the National CI —
From Cloud Prototyping to HPC Training to Large-Scale HTC Inference with
Jetstream2, Anvil, and the Open Science Pool."*

The tutorial walks through a complete, end-to-end machine learning workflow
across three national cyberinfrastructure resources:

#. **Data exploration & preprocessing** on Jetstream2 (cloud).
#. **Model training** on Purdue Anvil (HPC).
#. **Large-scale inference** on the OSPool (HTC).

Building the docs locally
-------------------------

The documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_::

    pip install -r docs/requirements.txt
    cd docs
    make html

Then open ``docs/build/html/index.html`` in a browser.

The published version is built automatically by Read the Docs from
``docs/source/conf.py``.
