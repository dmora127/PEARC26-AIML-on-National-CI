Introduction
============

Motivation

   This tutorial provides hands-on experience running AI workflows across three major computational paradigms: cloud computing, high-performance computing (HPC), and high-throughput computing (HTC). As artificial intelligence research continues to grow in scale and complexity, researchers increasingly need to understand how to leverage the strengths of different computing environments to efficiently train models, process data, and execute large-scale workflows. Throughout the workshop, participants will explore the advantages, limitations, and ideal use cases of each paradigm, gaining practical insight into how to select and optimize computational resources for their own research needs. The skills gained here will help reesearchers better understand how to balance scalability, cost, performance, and flexibility for their own research.
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

.. An Introduction::

   Cloud computing: Cloud computing provides on-demand access to computing resources such as virtual machines, GPUs, storage, and software services over the internet. Resources can be provisioned and scaled dynamically, making cloud environments particularly valuable for interactive development, rapid prototyping, burst workloads, and services that require flexibility. Researchers can create highly-customized computing environments to address the needs of different workflows. 
   High Performance Computing (HPC): High Performance Computing combines the power of many compute nodes using high-speed networks and parallel filesystems to solve complex computational problems. HPC is best suited for tightly coupled workloads where many processors must work together on a single problem, such as large-scale simulations and AI model training. The primary goal of HPC is to reduce the time required to complete computationally intensive tasks. 
----> Traditionally, HPC systems are limtied by 
   High Throughput Computing (HTC): High Throughput Computing focuses on maximizing the total amount of computational work completed over time rather than accelerating a single job. HTC environments are designed to run large numbers of independent or loosely coupled tasks, making them ideal for parameter sweeps, Monte Carlo simulations, image processing, and inference workflows. By distributing many jobs across available resources, HTC enables researchers to efficiently process workloads that may consist of thousands or millions of individual computations.

Cloud (Jetstream2)
~~~~~~~~~~~~~~~~~~
Cloud computing provides on-demand access to computing resources that can be rapidly provisioned and scaled as needed. Unlike HPC, which is optimized for tightly coupled workloads, anddHTC, which aggregates distributed resources for large numbers of independent jobs, cloud platforms emphasize flexibility, elasticity, and self-service access to compute, storage, networking, and specialized services.

Cloud environments allow researchers to create customized computing infrastructure, selecting the operating systems, software stacks, hardware configurations, and accelerator resources (e.g., GPUs) best suited to their workloads. Resources can be scaled up or down as requirements change, making cloud computing well suited for prototyping and exploring resources in AI/ML workflows. 

While cloud platforms can support both HPC and HTC workloads, they typically rely on virtualized resources such as virtual machines (VMs). This provides greater flexibility and virtually unlimited scalability, but may come with less specialized performance than dedicated HPC systems for tightly coupled applications and fewer parallel running jobs than HTC systems.

HPC (Purdue Anvil)
~~~~~~~~~~~~~~~~~~
High Performance Computing (HPC) systems excel at running tightly coupled computational workflows, where many processes must communicate frequently and coordinate closely while running. These workloads require specialized infrastructure to efficiently exchange data and synchronize work across multiple compute nodes.

HPC environments combine technologies such as InfiniBand networking for low-latency communication, MPI and OpenMP for parallel programming, and high-performance file systems such as VAST and Lustre for fast data access. Together, these components allow a single job to utilize resources from multiple nodes, including thousands of CPU cores and large numbers of GPUs.

Common HPC schedulers include Slurm, PBS, and Moab. Purdue University's Anvil cluster uses the Slurm Workload Manager to allocate resources and schedule jobs.

HPC systems are particularly well suited for large-scale simulations and AI/ML training, where applications benefit from access to large amounts of compute power, memory, and accelerator resources. However, HPC capacity is limited by the amount of hardware that can fit within a single data center (unlike HTC) and is generally less flexible than cloud environments for supporting highly diverse workloads.

HTC (OSPool)
~~~~~~~~~~~~
High Throughput Computing (HTC) systems excel at running loosely coupled computational workflows, where many independent tasks can execute with little or no communication between them. Rather than focusing on a single large job, HTC emphasizes completing large numbers of jobs efficiently across a distributed pool of resources.

HTC environments use technologies such as HTCondor for job scheduling, distributed data services, and software containers to run workloads across clusters, cloud resources, and institutions around the world. This allows researchers to access computing capacity that far exceeds what is available at a single site or datacenter. 

obs may execute on systems with different processor architectures, operating systems, hardware configurations, and software environments. As a result, applications must be designed to run reliably across a diverse set of resources.
On platforms such as the OSPool, jobs are matched to available resources contributed by institutions around the world. Because resource availability varies over time and jobs may be preempted or restarted, exact runtimes are not guaranteed. However, HTC excels at running massively parallel workloads—such as AI inference, parameter sweeps, and large-scale data processing—that can take advantage of a large, heterogeneous pool of resources to significantly reduce overall time to completion.

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
