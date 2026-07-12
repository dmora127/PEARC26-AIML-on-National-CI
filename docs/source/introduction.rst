Introduction
============

.. Motivation::

   This tutorial provides hands-on experience running AI workflows across three major computational paradigms: cloud computing, high-performance computing (HPC), and high-throughput computing (HTC). As artificial intelligence research continues to grow in scale and complexity, researchers increasingly need to understand how to leverage the strengths of different computing environments to efficiently train models, process data, and execute large-scale workflows. Throughout the workshop, participants will explore the advantages, limitations, and ideal use cases of each paradigm, gaining practical insight into how to select and optimize computational resources for their own research needs.
   Participants will gain direct experience using representative platforms from each ecosystem: Jetstream2 (cloud), Anvil (HPC), and OSPool (HTC). Together, these systems illustrate a diverse set of computational resources available at no cost to U.S.-based researchers through national cyberinfrastructure programs. By the end of the tutorial, attendees will have a stronger foundation for designing scalable, efficient AI workflows that take advantage of the most appropriate computing resources for each stage of their work.


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
