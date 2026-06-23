Running Parts 1–3 with HTCondor Annexes
=======================================

The :doc:`orchestration-model` page made the case for HTCondor Annexes: they let
you fold capacity you already have an allocation on — Jetstream2, Anvil, a local
workstation — into the pool you submit to. This page is the hands-on companion.
It covers what an annex actually is, how to stand one up in self-service mode,
and how to point each of the Part 1–3 stages at the right annexed resource.

.. note::

   The self-service annex workflow shown here uses a newer HTCondor (26.0)
   release that is not yet public. Commands and file names may shift before the
   tutorial; treat this as a working draft and verify against the running
   version during the live session.

What Is an HTCondor Annex?
--------------------------

An annex provisions machines from an external resource and starts HTCondor
**execute points** (EPs) on them that report back to your **access point** (AP).
Once those EPs report back to your AP, they're indistinguishable from any other
execute point: HTCondor matches your queued jobs onto them automatically, moves
files out and results back, and cleans up when the work is done. You keep
submitting exactly as before — the annex just widens where that work can land.

.. todo:: Concept diagram: AP + queued jobs on one side; a Slurm allocation launching EPs that "phone home" to the AP on the other.

Setting Up a Self-Service Annex
-------------------------------

Prerequisite: enable the feature on your access point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The access point must have the HPC annex feature enabled in its configuration:

.. code-block:: text

   use feature: hpc_annex

Create and launch the annex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The self-service workflow keeps job submission and annex creation as separate
steps. You first queue jobs that *target* a named annex, then create the annex
that will run them.

#. **Submit jobs targeting the annex.** The job sits idle until matching annex
   capacity appears.

   .. code-block:: console

      $ htcondor job submit my_job.sub --annex-name NAME

#. **Create the annex.** The only required argument is the name. This produces a
   tar file plus transfer instructions.

   .. code-block:: console

      $ htcondor annex create NAME

    This will generate a tarball (``annex-setup.tar.gz``) containing the launch scripts and
    configuration for the annex. It will also print instructions for transferring the tarball to the Slurm cluster and running the setup script.

#. **Transfer the tar to the Slurm cluster** and, on the login node, extract it.
   Extraction creates a subdirectory containing a ``README`` and the launch
   files.

#. **Run the setup script** from inside that subdirectory:

   .. code-block:: console

      $ cd <extracted-subdir>
      $ ./annex-setup.sh

   This generates ``hpc.slurm``. The only Slurm options preset for you are the
   stdout/stderr file names.

#. **Edit** ``hpc.slurm`` to add the ``#SBATCH`` options for the allocation you
   want (nodes, cores, time, partition, etc.), then submit it:

   .. code-block:: console

      $ sbatch hpc.slurm

Once the Slurm job starts, it launches the HTCondor execute points, which join
the pool and begin running the jobs you queued against ``NAME``.

What happens at runtime
~~~~~~~~~~~~~~~~~~~~~~~~~

Inside the Slurm allocation, the annex lays out a couple of working
directories:

- ``pilot.<jobid>/`` — Condor binaries, configuration, and the execute
  directory. **Removed** after the daemons exit.
- ``annex-logs.<jobid>/`` — Condor ``LOG`` directories. **Not** cleaned up, so
  they remain available for debugging after the allocation ends.

The execute points auto-detect the CPUs, memory, and runtime that are allocated
and stay within those limits, so they won't try to hand your jobs more than the
allocation actually granted.

Customizing the annex
~~~~~~~~~~~~~~~~~~~~~~~

Three optional files in ``~/.condor/`` on the HPC system let you adapt the annex
to the cluster's conventions:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - File
     - Scope
   * - ``annex_config``
     - Sourced by ``annex-setup.sh``. Use it to load modules and set
       ``SCRATCH`` — i.e. where ``pilot.<jobid>`` lives, such as another volume
       or node-local disk.
   * - ``annex_slurm_args``
     - Inserted into ``hpc.slurm`` after the always-set ``#SBATCH`` lines. Use
       it for extra ``#SBATCH`` options plus any commands to run inside the
       Slurm job before the execute point starts.
   * - ``annex_pilot_config``
     - Dropped into the execute points' ``config.d`` directory to override
       Condor configuration on the borrowed machines.

With the mechanics in hand, the rest is a matter of pointing each stage at the
resource that fits it. Each subsection below maps one Part 1–3 stage onto an
annex; the differences come down to *how* the capacity is provisioned — a cloud
VM, a Slurm allocation, or a machine you already control.

Annexing Cloud Capacity (Jetstream2)
------------------------------------

Part 1's preprocessing — decoding recordings and rendering spectrograms — is
CPU-bound, interactive-style work that maps well onto a cloud VM. Annexing
Jetstream2 means provisioning an instance and having it run an execute point
that joins your pool, so the ``preprocess`` stage lands there instead of on
generic OSPool capacity.

Unlike the Slurm flow above, there's no scheduler to submit to: you bring up the
VM yourself (Exosphere, the Horizon dashboard, or the OpenStack CLI), then start
an EP on it that reports back to your access point.

.. todo::

   Pin down the concrete steps: which Jetstream2 image/flavor to launch, how the
   EP is installed and pointed at the AP (annex token vs. ``condor_annex``-style
   provisioning), and how to size the instance for the Part 1 workload. Verify
   against the running HTCondor version.

.. todo::

    Explain why we would use an annex for Jetstream2 instead of just running the preprocess stage on a Jetstream2 VM directly. Explain when we would want to use Jetstream2 directly instead of an annex, and when we would want to use an annex instead of Jetstream2 directly.

Annexing HPC Capacity (Anvil)
-----------------------------

Part 2's GPU training is the natural fit for an HPC annex: it needs a real GPU
for a bounded stretch of time, which is exactly what an Anvil Slurm allocation
provides. This is the self-service flow from `Setting Up a Self-Service Annex`_
applied directly — create the annex, move the tar to Anvil, and submit
``hpc.slurm`` requesting a GPU node.

The ``#SBATCH`` options you'd normally put in the Part 2 batch script go into
``hpc.slurm`` (or, to make them permanent, into ``~/.condor/annex_slurm_args``).
For the tutorial's training stage that's an account, the GPU partition, and one
GPU:

.. code-block:: text

   #SBATCH --account cis260991
   #SBATCH --partition=gpu
   #SBATCH --gpus-per-node=1
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=20G
   #SBATCH -t 01:30:00

Once the allocation starts, the EPs it launches advertise the GPU, and the
``train`` job — which already requests a GPU — matches onto them. The walltime
you request bounds how long the annex lives, so size it to cover the training run
with a little padding.

.. todo::

   Show the end-to-end commands for this specific case (``htcondor annex
   create``, transfer, ``annex-setup.sh``, ``sbatch``) and confirm whether the
   GPU request belongs in ``hpc.slurm`` directly or in ``annex_slurm_args``.

Annexing Your Local Capacity
-----------------------------

You can also annex capacity you already control — a lab workstation, a
departmental cluster, or a local institutional HPC system. The same model
applies: start an execute point on the machine, point it at your access point,
and it becomes eligible to run your jobs. This is a good home for Part 3's
inference shards when you have idle local hardware and would rather not wait on
shared pool capacity.

For a single workstation or a handful of machines, this is just installing and
configuring an EP; for a local Slurm cluster, it's the same self-service flow as
Anvil pointed at your own scheduler.

.. todo::

   Document the local-EP path: the configuration that joins a machine to
   your pool, and the minimum HTCondor install required on the execute side.

Tying the Stages Together
-------------------------

Annexes decide *where* each stage can run; the DAG in :doc:`orchestration-model`
decides *when*. Put together, the running example routes cleanly across the
national CI:

.. list-table::
   :header-rows: 1
   :widths: 16 28 56

   * - Stage
     - Annexed resource
     - Why it lands there
   * - ``preprocess``
     - Jetstream2 annex
     - CPU-bound, interactive-style cloud work
   * - ``train``
     - Anvil annex (GPU)
     - needs a real GPU for a bounded run
   * - ``infer``
     - OSPool (or a local annex)
     - embarrassingly parallel, high-throughput

Each node's submit file targets its annex by name; the DAG's edges hand each
stage's output to the next (spectrograms + metadata → checkpoint → predictions).
The result is the payoff promised in :doc:`orchestration-model`: one workflow,
submitted once, running across cloud, HPC, and HTC without being rewritten for
any of them.

.. todo::

   Once the per-resource annex commands are pinned down, add the exact
   submit-file lines that bind ``preprocess`` and ``train`` to their annex names,
   and cross-check against the ``pipeline.dag`` example.
