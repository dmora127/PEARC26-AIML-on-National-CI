Packaging the Model & Code
==========================

.. todo::

   Make each job self-contained: bundle the model, inference script, and
   dependencies (container or portable environment) so jobs run anywhere on
   the pool. Plan how inputs and outputs are transferred.

The Inference Script
--------------------

.. todo:: A script that loads the model and runs inference on a shard of inputs.

Dependencies & Containers
-------------------------

The inference script needs PyTorch, torchvision, and a handful of supporting
libraries (pandas, Pillow, NumPy) to load the checkpoint and process
spectrograms. On a single machine you'd just install those into a virtual
environment and forget about it — but the OSPool is not a single machine. It's a
*heterogeneous* pool stitched together from hundreds of contributing institutions,
so your job might land on almost any Linux distribution, with different system libraries
and no say over what's installed. You don't have administrator access to those execute
nodes, and there's no guarantee the version of Python — let alone PyTorch — that
you need is present. For results to be reproducible, every one of your jobs has
to run against the *same* software environment regardless of where it lands.

The reliable way to guarantee that is to ship the environment with the job inside
a **container**.

.. figure:: https://chtc.cs.wisc.edu/images/container-analogy-infographic1.png
    :alt: A camping backpack with various camping supplies backed into it, including pots, utensils, and a flashlight.
    :align: center
    :width: 50%

    As an analogy, you could consider a container to be like a camping backpack. Every time you plan to use it, you will need a standard set of gear, which you could pre-pack. Other items, like maps, food, or fuel would depend on where you’re going, but you would still have access to the standard gear.

How containers work
^^^^^^^^^^^^^^^^^^^

A container bundles an entire user-space operating system — Python, your
libraries, their system dependencies, all pinned to specific versions — into a
single portable image. When a job runs inside a container, it sees that packaged
environment instead of whatever happens to be installed on the execute node, so
the same image produces the same environment on every machine in the pool. The
software is installed *once*, when you build the image; jobs never install
anything at run time, which is what makes them both fast to start and
reproducible.


.. figure:: https://chtc.cs.wisc.edu/images/container-analogy-infographic2.png
    :alt: A camping backpack with various camping supplies backed into it, including pots, utensils, and a flashlight.
    :align: center
    :width: 50%

    A “container image” is the persistent, on-disk copy of the container. When building the container you can choose the operating system you want to use, and can install programs as if you were the owner of the computer. When a container is “running” or “executed”, the container image is used to create the run time environment for executing the programs installed inside of it.

On the OSPool we use `Apptainer <https://apptainer.org/>`_ (formerly Singularity),
which packages a container as a single ``.sif`` file. Because it's just a file,
it moves through the same data-transfer machinery as the rest of your job inputs
— including the OSDF — so it's easy to stage close to wherever your jobs run.

Building the container from a definition file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You describe the environment you want in an Apptainer *definition file*. It
starts from an existing base image (here, an official Python image from Docker
Hub) and lists the build steps in a ``%post`` section. Save the following as
``inference.def``:

.. code-block:: singularity

    Bootstrap: docker
    From: python:3.11-slim

    %post
        pip install --no-cache-dir \
            torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install --no-cache-dir pandas pillow numpy

    %environment
        export PYTHONUNBUFFERED=1

    %labels
        Description BirdCLEF inference environment (PyTorch CPU)

A few things worth noting:

- ``Bootstrap``/``From`` pick the starting image — a slim Python 3.11 base.
- The ``%post`` section runs at *build* time and is where software gets
  installed. We pull the CPU-only PyTorch wheels, which keeps the image small;
  batch inference here is massively parallel across many ordinary cores, so
  it doesn't need a GPU.
- ``%environment`` sets variables that take effect every time the container runs.

Build the ``.sif`` image from the definition file with:

.. code-block:: shell

    apptainer build inference.sif inference.def

When it finishes you'll see ``INFO: Build complete: inference.sif``. Building
requires more memory and disk than a login session allows, so run the build
inside an interactive job on the access point rather than on the login node, then
copy the finished image into your OSDF-backed data area:

.. code-block:: shell

    cp inference.sif /ospool/ap40/data/<username>/

.. tip::
    Before scaling out, start an interactive session inside the image
    (``apptainer shell inference.sif``) and confirm the packages import — e.g.
    ``python3 -c "import torch, torchvision; print(torch.__version__)"``. Catching
    a missing dependency now is far cheaper than discovering it across a thousand
    queued jobs.

Using the container in your jobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the image is staged, running a job inside it takes a single line in the
submit file. Point ``container_image`` at the staged ``.sif`` via an ``osdf://``
URL so each job pulls it from the data federation into its sandbox:

.. code-block:: text

    container_image = osdf:///ospool/ap40/data/<username>/inference.sif

    executable = run_inference.sh
    # ... transfer_input_files, arguments, requests, queue ...

With ``container_image`` set, HTCondor fetches the image, starts the job inside
it, and runs your ``executable`` within the container environment — so the
``python3`` your inference script calls is the one you built, every time, on
every node. The rest of the submit file (input/output transfer, resource
requests, and how the workload is split across jobs) is covered in
:doc:`submit-file` and :doc:`scaling-out`.

Data Movement
-------------

The container solves *software* portability, but each job still needs its
*data*: the model checkpoint, the slice of inputs it's responsible for, and a
place to put its results. On the OSPool that's something you have to plan for
explicitly, because **there is no shared filesystem**. An execute node halfway
across the country can't reach the files on your access point, so every job has
to bring its inputs along with it and hand its outputs back when it finishes.
HTCondor manages both directions through directives in the submit file — nothing
is implicitly available the way it would be on a single shared machine.

Two ways to move data
^^^^^^^^^^^^^^^^^^^^^^

There are two transfer mechanisms, and the right one depends mostly on file size,
frequency of file modifications, and on how many jobs need the same file.

**Standard HTCondor file transfer** is the default. You list inputs with
``transfer_input_files`` and HTCondor copies them from your access point into the
job's sandbox; outputs are returned automatically (or named explicitly with
``transfer_output_files``). This is the simplest path and the right choice for
per-job files up to roughly **1 GB each**, with a similar cap on the total moved
per job. It pulls straight from the access point, though, so it doesn't scale
well when thousands of jobs all want the *same* file at once.

.. code-block:: shell

    transfer_input_files = /home/jane.doe/myPythonScript.py

**Using the OSDF (Powered by Pelican)** is the alternative, addressed with ``osdf://`` URLs that
point at your staging area under ``/ospool/ap40/data/<username>/``. The data
federation caches objects near the execute nodes, so it's the right choice for:

- files **1 GB or larger**, which standard transfer isn't meant to carry;
- anything **shared across many jobs**, regardless of size — the caches absorb
  the load instead of every job hammering your access point;
- **container images**, which is exactly why the ``.sif`` from the previous
  section is referenced with an ``osdf://`` URL.

In order to use the OSDF, files must be placed in an OSDF-accessible object storage/directory
exposed to the OSDF via Pelican. On the OSPool, each user is given a directory, ``/ospool/ap40/data/<username>/``,
which allows files within it to be accessible to the OSDF. To use this directory with the OSDF,
simply store your data under ``/ospool/ap40/data/<username>/`` and add ``osdf:///ospool/ap40/data/<username>/<filename>``
to your ``transfer_input_files`` line in your submit file. For example:

.. code-block:: shell

    transfer_input_files = osdf:///ospool/ap40/data/jane.doe/myInputData.tar.gz

The OSDF should be avoided for frequently changed objects, such as log files, as Pelican treats all objects
as immutable. Once a file has been requested, copies of it linger in caches near the execute nodes, and a
later request for the same path may be served that cached copy rather than the version currently sitting in
your staging directory. Overwriting a file in place therefore does *not* reliably update what jobs receive:
the change may not propagate until the cached copy expires, and in the meantime different jobs can end up with
different versions. This is exactly why frequently changing files should come back through
standard HTCondor file transfer instead. If you do need to redistribute an updated input through the OSDF, give
it a new name (for example, a version suffix or date) so jobs request a fresh path that no cache has seen.

.. tip::

    You can use Pelican transfers to put data in other Pelican-backed namespaces. For example, your local cluster
    or ACCESS-CI resource may have a Pelican-backed object store you have access to. You can use this ability to transfer
    files directly from your job to another object store, without passing through the AP. Similarly, you can use Pelican
    to pull objects from other object stores within the federation. For more information on some of these object stores,
    visit: https://osg-htc.org/services/osdf/data

.. note::
   Whichever mechanism you use, jobs are always *submitted* from your ``/home``
   directory on the access point. Large or shared inputs live in your
   ``/ospool/ap40/data/`` staging area and are pulled in via the OSDF; ``/home``
   is just where the submit file and small per-job files sit.

Our transfer plan
^^^^^^^^^^^^^^^^^

That maps cleanly onto the four kinds of file this workload moves:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Artifact
     - Method
     - Why
   * - ``inference.sif`` container
     - OSDF
     - A large image needed by every job — the canonical OSDF case.
   * - ``best_efficientnet_b0.pt`` checkpoint
     - OSDF
     - Only a few tens of MB, but *every* job loads the same copy, so the
       caches keep the access point out of the loop.
   * - This job's input file
     - HTCondor
     - A small, per-job list of recordings to classify — different for each
       job and well under the size limit.
   * - Prediction output (CSV)
     - HTCondor
     - A small per-job result that comes straight back to the access point.

In submit-file terms that's the ``container_image`` line from the previous
section plus a few transfer directives:

.. code-block:: text

    # Shared, pulled from the OSDF so the caches absorb the load:
    container_image      = osdf:///ospool/ap40/data/<username>/inference.sif
    transfer_input_files = osdf:///ospool/ap40/data/<username>/best_efficientnet_b0.pt, shards/shard_$(Process).txt

    # Small per-job result returned to the access point automatically:
    transfer_output_files = predictions_$(Process).csv

Here the checkpoint is pulled from the OSDF while the per-job shard
(``shard_$(Process).txt``) comes along by standard transfer from ``/home``, and
the single output CSV is handed back the same way. How the input list is split
into those per-job shards — and how ``$(Process)`` fans a single submit file out
across the whole pool — is the subject of :doc:`scaling-out`.

.. note::

    ``container_image`` will automatically include your container image in the transfer_input_files for your job. We recommend
    using the ``osdf:///`` to transfer your containers into your job's sandbox.