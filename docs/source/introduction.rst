Introduction
============

.. todo::

   Set the stage: motivate why a single AI/ML project benefits from spanning
   multiple national CI resources rather than running everything in one place.

Learning Objectives
-------------------

By the end of this tutorial, participants will be able to:

- Describe the strengths of cloud, HPC, and HTC resources and when to use each.
- Interactively explore and preprocess a dataset on a Jetstream2 cloud VM.
- Submit and monitor a GPU training job on the Purdue Anvil HPC cluster.
- Run large-scale, high-throughput inference across the OSPool with HTCondor.
- Move data and artifacts (datasets, models, results) between these systems.

The Three Paradigms
-------------------

.. todo::

   Briefly contrast Cloud vs. HPC vs. HTC. One short subsection each.

Cloud (Jetstream2)
~~~~~~~~~~~~~~~~~~

.. todo:: Interactive, on-demand VMs; great for prototyping and exploration.

HPC (Purdue Anvil)
~~~~~~~~~~~~~~~~~~

.. todo:: Tightly-coupled, GPU-accelerated batch jobs via Slurm; great for training.

HTC (OSPool)
~~~~~~~~~~~~

.. todo:: Many independent jobs via HTCondor; great for massively parallel inference.

The Example Application
-----------------------

.. todo::

   Describe the dataset and model used throughout the tutorial so each section
   has a consistent running example.

Architecture Overview
---------------------

.. todo::

   Add a diagram showing data/artifact flow: dataset → Jetstream2
   (explore/preprocess) → Anvil (train) → OSPool (infer) → results.
