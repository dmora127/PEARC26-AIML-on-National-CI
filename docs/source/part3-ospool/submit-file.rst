Writing the HTCondor Submit File
================================

With the container built and the model staged, the last piece is telling
HTCondor *what to run and what it needs*. You do that in a **submit file** — a
short plain-text file that describes a job (or a whole set of jobs) as a list of
``keyword = value`` pairs. When you hand it to ``condor_submit``, HTCondor reads
that description, finds machines in the pool that match, and dispatches the work.

A submit file pulls together the three concerns from the previous pages into one
place. It declares:

- **What to run** — the executable and the container it runs inside.
- **What resources it needs** — how many CPUs, how much memory and disk, and any
  operating-system or GPU requirements.
- **What files move and where output goes** — the inputs transferred in, the
  results sent back, and the ``output``, ``error``, and ``log`` files that
  capture what happened.

The strategy for the rest of Part 3 is to **get one job right before scaling to
many.** A single job is far easier to debug — you can read its logs, confirm the
container loads the model, and check that predictions come back in the form you
expect. Once that one job is solid, turning it into hundreds is a small change to
the submit file, which is the subject of :doc:`scaling-out`. This page builds
that first single-job submit file and walks through requesting resources for it.

A Single-Job Submit File
------------------------

Let's start with a submit file that runs one job on one recording. The following
example is a complete submit file for the BirdCLEF PEARC26 inference run. It
assumes the container and model checkpoint are already staged in the Open Science Data
Federation (OSDF) and that the recording to process is ``ES_01_20230115_060000.ogg``.

.. code-block:: bash

    container_image = osdf:///osg-public/containers/birdclef-pearc26.sif
    executable = run_inference.py

    transfer_input_files = osdf:///osg-public/pearc26-aiml/data/<FILE>, best_efficientnet_b0.pt

    output = logs/infer_$(Cluster)_$(Process).out
    error = logs/infer_$(Cluster)_$(Process).err
    log = logs/infer_$(Cluster)_$(Process).log

    request_cpus = 1
    request_gpus = 1
    request_memory = 1 GB
    request_disk = 10 GB

    minimum_gpu_memory = 39 GB

    queue


Submit Your First Test Job
-----------------

You can submit this job to the pool with:

.. code-block:: bash
    condor_submit run_inference.sub


