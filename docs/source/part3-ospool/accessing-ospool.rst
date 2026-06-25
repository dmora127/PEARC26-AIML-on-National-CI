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
around making a *single* job possible running across various tightly coupled nodes of
CPUs and GPUs, a fast interconnect, and a shared filesystem, all reserved for you
for the duration of the run. It's the right tool when the pieces of your
computation have to talk to each other constantly and require highly specialized configurations
or very large numbers of cores across multiple nodes (using MPI).

High-Throughput Computing (HTC), which the OSPool provides, optimizes for
something else: getting through a *large number of independent jobs* in the
aggregate. Instead of one reserved supercomputer, the OSPool is a distributed,
**opportunistic** pool — machines contributed by dozens of institutions across
the country, many of them filling in idle time on hardware owned by someone else.
You don't reserve a node and keep it; you submit a list of jobs and the system
runs each one wherever and whenever a slot frees up.

That model has a few consequences that shape how we package the inference work:

- **The work must be divisible into independent pieces.** Our task qualifies
  naturally — classifying one recording never depends on the result of another,
  so the input set can be split into shards and each shard handled by its own
  job. Problems like this are called *massively parallel* (or *embarrassingly parallel*).
- **Each job must be self-contained.** A job can land on almost any machine, so
  it has to carry everything it needs — software *and* data — with it. That's why
  the previous page built a :doc:`container <packaging>` for the environment and
  planned explicit file transfer for the model and inputs; there is no shared
  filesystem to fall back on.
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

Replace ``<username>`` with your OSG username. Access points authenticate with either a
registered **SSH key** or your institutional Single-Signon login. While the single-signon link can be convinient for new
users, we recommend generated a key pair and added the public key to your OSG account.
depending on the access point you may also be prompted for a one-time code from a
multi-factor authentication app. If you haven't set this up yet, follow the
`OSG account setup guide <https://portal.osg-htc.org/documentation/overview/account_setup/registration-and-login/>`_.

.. tip::
    If you're following along as part of the in-person workshop, your OSG account
    and SSH access have already been provisioned for you — use the username and
    access point you were given at the start of the session.

A couple of habits to carry in from the start:

- **Submit from your home directory.** ``/home`` on the access point is where
  submit files and small per-job files live. Large or shared inputs — the model
  checkpoint and the container image you staged in :doc:`packaging` — belong in
  your ``/ospool/ap40/data/<username>/`` area and are pulled in via the OSDF.
- **Don't run heavy compute on the access point itself.** It's a shared,
  submit-only machine — running the actual inference there (rather than shipping
  it out as jobs) slows it down for everyone. Quick tests are fine; the real work
  goes to the pool.

Once you can log in and see your staged model and container, you're ready to
write the submit file that turns the inference script into a pool of jobs, which
is the subject of :doc:`submit-file`.
