AI/ML Workflows on the National CI
==================================

**From Cloud Prototyping to HPC Training to Large-Scale HTC Inference with
Jetstream2, Anvil, and the Open Science Pool**

Welcome to the companion guide for the PEARC26 tutorial. This guide walks
through a complete, end-to-end AI/ML workflow that spans three complementary
national cyberinfrastructure (CI) resources, using the right tool for each
stage of the machine learning lifecycle:

#. **Prototype** — interactively explore and preprocess data on a cloud VM
   using `Jetstream2 <https://jetstream-cloud.org/>`_.
#. **Train** — run accelerated model training on the
   `Purdue Anvil <https://www.rcac.purdue.edu/anvil>`_ HPC cluster.
#. **Infer at scale** — fan out high-throughput batch inference across the
   `OSPool <https://osg-htc.org/services/open_science_pool.html>`_.

.. note::

   This guide is under active development for the PEARC26 tutorial. Follow
   along with the live session or work through it at your own pace.

.. tip::

   Complete :doc:`setup/index` before the hands-on sections — in particular,
   register for your accounts at least 24 hours ahead so provisioning finishes
   in time.

The Workflow at a Glance
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Stage
     - Resource
     - Paradigm
     - What you'll do
   * - Data Exploration & Preprocessing
     - Jetstream2
     - Cloud / interactive
     - Spin up a VM, explore the dataset, build the preprocessing pipeline
   * - Model Training
     - Purdue Anvil
     - HPC / batch (Slurm)
     - Request GPUs, train the model, checkpoint results
   * - Large-Scale Inference
     - OSPool
     - HTC (HTCondor)
     - Package the model and fan out inference across many jobs

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   setup/index