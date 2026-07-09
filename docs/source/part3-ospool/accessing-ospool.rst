Accessing the OSPool
====================

In Part 2 you trained the model on Anvil, where you asked the scheduler for a
single powerful node — one GPU, for a fixed block of time — and ran one job on
it. Inference is a different shape of problem: you have a large set of
recordings to classify, and each one is independent of the others. That's a poor
fit for a single big machine and an excellent fit for the **OSPool**, where the
work is spread across thousands of ordinary machines running many small jobs at
once. This page covers the mindset shift that comes with that — *high-throughput*
rather than *high-performance* computing — and how to get onto an access point to
start submitting.

The HTC Mental Model
--------------------

High-Performance Computing (HPC), like the Slurm cluster on Anvil, is built
around making a *single* large job possible; High-Throughput Computing (HTC),
which the OSPool provides, optimizes for getting through a *large number of
independent jobs* in the aggregate. The two models differ at every level:

.. list-table::
   :header-rows: 1

   * -
     - HPC (Anvil, Slurm)
     - HTC (OSPool, HTCondor)
   * - Optimized for
     - One large, tightly-coupled computation
     - Many independent jobs, completed in the aggregate
   * - The hardware
     - Tightly coupled nodes with a fast interconnect and a shared filesystem
     - A distributed, **opportunistic** pool — machines at dozens of
       institutions contributing idle time
   * - Getting resources
     - You reserve nodes; they're yours for the duration of the run
     - No reservation — each job runs wherever and whenever a slot frees up
   * - The right tool when
     - Pieces that talk to each other constantly, need specialized
       configurations, or need many cores across nodes (MPI)
     - Work that divides into independent pieces, each with its own smaller
       requirements, that can run anywhere

    .. image:: ../assets/ospool/conceptual-htc-hpc.png
    :alt: Conceptual diagram contrasting HPC and HTC
    :width: 80%
    :align: center

That model has a few consequences that shape how we package the inference work:

- **The work must be divisible into independent pieces.** Our task qualifies
  naturally — classifying one recording never depends on the result of another,
  so the input set can be split into shards and each shard handled by its own
  job. Problems like this are called *massively parallel* (sometimes poorly refer to as *embarrassingly parallel*).
- **Each job must be self-contained.** A job can land on almost any machine, so
  it has to carry everything it needs — software *and* data — with it.
- **Jobs should be modest and resilient.** Because resources are opportunistic, a
  job can occasionally be interrupted and rescheduled elsewhere. Many small,
  short jobs ride this out far better than a few enormous ones — if one is
  evicted, only a little work is lost and it simply runs again somewhere else.

The payoff is scale. Rather than funneling the whole batch through one reserved
node, you can have *hundreds* of jobs running concurrently across the pool — so
the more you can break the work into independent pieces, the faster the entire
batch finishes. For an massively parallel task like ours, that breadth is
exactly the resource that matters.

Logging Into an Access Point
----------------------------

You interact with the OSPool through an **access point** — a login node where you
stage files, write submit files, and run ``condor_submit``. The access point is
*not* where your jobs run; it's the front door from which HTCondor dispatches them
out to the pool. For this tutorial we use the ``AP40`` access point, which you reach
over SSH:

.. code-block:: shell

    ssh <username>@ap40.uw.osg-htc.org

Replace ``<username>`` with your OSPool username. Access points authenticate with either a
registered **SSH key** or your institutional Single-Signon login. While the single-signon link can be convenient for new
users, we recommend generated a key pair and added the public key to your OSPool account.
depending on the access point you may also be prompted for a one-time code from a
multi-factor authentication app. If you haven't set this up yet, follow the
`OSPool account setup guide <https://portal.osg-htc.org/documentation/overview/account_setup/registration-and-login/>`_.

.. tip::
    If you're following along as part of the in-person workshop, your OSPool account
    and SSH access have already been provisioned for you — use the username and
    access point you were given at the start of the session.

A couple of habits to carry in from the start:

- **Submit from your home directory.** ``/home`` on the access point is where
  submit files and small per-job files live. Large or shared inputs — the model
  checkpoint and the container image you will stage in :doc:`packaging` — belong in
  your ``/ospool/ap40/data/<username>/`` area and are pulled in via the OSDF.
- **Don't run heavy compute on the access point itself.** It's a shared,
  submit-only machine — running the actual inference there (rather than shipping
  it out as jobs) slows it down for everyone.

Once you can log in and see your staged model and container, you're ready to
write the submit file that turns the inference script into a pool of jobs, which
is the subject of :doc:`submit-file`.
