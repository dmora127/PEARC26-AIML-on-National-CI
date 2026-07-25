Running on ACCESS-CI and NAIRR Pilot Allocations
================================================

Everything you ran in Parts 1–3 was charged to the tutorial's shared allocation
(CIS260991 — *Training: AI/ML Workflows Across National Cyberinfrastructure*),
which expires after the workshop. This page is about doing the same thing on
your own: the national programs that grant compute to U.S.-based researchers at
no cost, how to request time from each, and how to point the orchestration
layer — your access point and its annexes — at the resources a grant gives you.

The short version: **ACCESS-CI** allocates time on named resources like
Jetstream2 and Anvil; the **NAIRR Pilot** focused on AI-heavy awards (GPU time,
model and cloud credits, datasets) using some of the ACCESS resources plus other NAIRR-specific
ones; and the
**OSPool** requires no allocation at all — any eligible researcher can submit
from day one.

NSF ACCESS-CI Allocations
-------------------------

`ACCESS <https://access-ci.org/>`_ (Advanced Cyberinfrastructure Coordination
Ecosystem: Services & Support) is the NSF program — successor to XSEDE — that
coordinates allocations across the national resource providers, including both
systems you used in this tutorial: Jetstream2 (cloud) and Anvil (HPC). You do
*not* need NSF funding to apply. Faculty, research staff, and postdocs at
U.S.-based institutions can serve as PI, and graduate students can lead an
Explore-tier project with their advisor as co-PI.

Allocation tiers
~~~~~~~~~~~~~~~~

ACCESS awards **ACCESS credits**, a common currency you then exchange for time
on specific resources. The tiers differ mainly in how many credits you can
request and how much proposal-writing that takes:

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - Tier
     - Credit ceiling
     - Request
     - Review
   * - Explore
     - up to 400,000
     - abstract (~1 paragraph)
     - rolling, typically days
   * - Discover
     - up to 1,500,000
     - ~1-page proposal
     - rolling, typically days
   * - Accelerate
     - up to 3,000,000
     - ~3-page proposal
     - rolling merit review, allow a few weeks
   * - Maximize
     - no ceiling
     - full proposal
     - semi-annual cycles

For work at the scale of this tutorial — prototyping a pipeline, a course
project, a first pass over a new dataset — **Explore is the right tier**, and
it is deliberately low-friction: a short description of what you want to do,
reviewed on a rolling basis. Projects can be upgraded to a higher tier later
without starting over. Credit ceilings and review timelines are adjusted
occasionally, so check the
`current allocation opportunities <https://allocations.access-ci.org/opportunities>`_
before you write anything.

Requesting time and mapping credits to resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. **Create an ACCESS ID** at https://access-ci.org/register (you already did
   this in :doc:`../setup/account-registration`, using your institutional
   identity).
#. **Submit a request** at https://allocations.access-ci.org under the tier
   that fits. For Explore, this is the abstract plus a CV.
#. **Exchange credits for resource units** once the award is active. Credits
   convert to resource-specific units — Jetstream2 service units (SUs), Anvil
   CPU core-hours or GPU-hours, and so on — at published exchange rates in the
   allocations marketplace. You can split one award across several resources
   and re-balance the exchange later as your usage becomes clearer.
#. **Add users to the project.** Each person you add can then log into the
   allocated resources with their own ACCESS ID, exactly as you did during
   setup.

The tutorial allocation is a concrete example of this mapping: one ACCESS
project (CIS260991) exchanged credits onto two resources. On Jetstream2 it
appears as an allocation you select in Exosphere when launching an instance;
on Anvil it becomes the Slurm account ``cis260991`` that batch jobs charge
against. Your own project will follow the same pattern under its own project
ID.

.. tip::

   GPU units are far more expensive in credits than CPU units, and cloud SUs
   burn for as long as an instance exists, not just while it computes. When
   sizing an Explore request, estimate the GPU-hours for training separately
   from the CPU time for everything else — the split usually matters more than
   the total.

NAIRR Pilot Allocations
-----------------------

The `NAIRR Pilot <https://nairrpilot.org/>`_ (National Artificial Intelligence
Research Resource Pilot) is an NSF-led, multi-agency effort — with DOE and a
broad set of agency and industry partners — to make AI research infrastructure
available beyond the handful of institutions that own large GPU clusters. Where
ACCESS allocates general-purpose research computing, the NAIRR Pilot is
AI-specific, and its catalog includes things ACCESS credits can't buy:

- **GPU compute** on national systems (many of them the same providers you met
  through ACCESS — Anvil's GPU partition among them — plus DOE and other
  agency resources).
- **Commercial cloud and model credits** contributed by industry partners,
  such as cloud GPU capacity and API credits for hosted foundation models.
- **Datasets, model repositories, and software platforms** for AI research.
- **NAIRR Classroom** allocations for using these resources in courses and
  training events.

Applying is comparable in effort to an ACCESS Discover/Accelerate request: a short project
description submitted through the `NAIRR Pilot portal
<https://nairrpilot.org/>`_, reviewed on a rolling basis for startup-scale
awards, with larger calls announced periodically. Eligibility mirrors ACCESS —
researchers and educators at U.S.-based institutions. Because it is a pilot,
the resource catalog and calls evolve; check the portal for what is currently
on offer rather than planning around a snapshot.

.. tip:: NAIRR offers Startup awards for getting started!

    NAIRR Startup awards are designed to get a project off the ground quickly,
    with a small amount of GPU time and cloud credits. They are ideal for
    prototyping a pipeline, testing a new dataset, or exploring a new model.
    These awards are typically reviewed on a rolling basis, and the application
    process is streamlined to minimize overhead. Application reviews typically
    take 1-3 weeks. If your project is in its early stages, a Startup award
    can provide the resources you need to demonstrate feasibility and gather
    preliminary results. Learn more about NAIRR Startup awards on
    the `NAIRR Pilot website <https://nairrpilot.org/>`_.

For the workflow in this tutorial, the natural use of a NAIRR award is the
``train`` stage: a NAIRR compute award on a GPU system takes the place of the
Anvil GPU allocation, and everything else about the pipeline stays the same.
If your award lands on an ACCESS resource provider, you'll manage accounts and
log in through the same ACCESS machinery you already know; the award simply
shows up as another project/account to charge against.

OSPool / PATh Facilities
------------------------

The third pillar needs the least paperwork: **the OSPool does not require an
allocation.** Any researcher affiliated with a U.S. academic, government, or
non-profit research institution can request an account on an OSG-operated access point
(the :doc:`registration you completed for this tutorial
<../setup/account-registration>`) and immediately submit jobs to the OSPool's
open capacity — tens of thousands of cores, and a meaningful number
of GPUs, contributed by campuses across the country. Usage is fair-share:
there is no bank of hours to exhaust, but also no guarantee of when capacity
arrives, which is exactly the trade described in
:doc:`Part 3 <../part3-ospool/index>`.

For workloads that need specialized resources rather than open capacity, the
`PATh facility <https://path-cc.io/services/credit-accounts/>`_ offers
dedicated CPU and GPU capacity through credit accounts, which **NSF-funded
projects** can request in their proposals or by contacting the PATh team.

Using Allocations with Annexes
------------------------------

The accounting rule for the annex model in
:doc:`../part3-ospool/annexes-across-ci` is simple: **allocation is spent where
the annex is provisioned, not where the jobs are submitted.** Submitting from
your AP is free; the Slurm job or cloud instance that launches the execute
points is what draws down an award. Concretely, per resource:

**Anvil (or any Slurm-scheduled HPC annex).** The allocation is named by the
``#SBATCH --account`` line in the ``hpc.slurm`` file the annex setup generates
— for the tutorial that was ``--account cis260991``; for your own project it
is your ACCESS project ID in lowercase, which ``mybalance`` on Anvil will list
along with your remaining hours. Put the account line in ``hpc.slurm``
directly, or in ``~/.condor/annex_slurm_args`` to make it permanent. The
charge is the Slurm allocation itself — nodes × walltime, in core-hours or
GPU-hours — and it accrues **whether or not HTCondor keeps the execute points
busy**, so size the requested walltime to the work you actually queued, with
modest padding.

**Jetstream2 (or any cloud annex).** The allocation is chosen when you launch
the instance — the Exosphere allocation selector you used in
:doc:`Part 1 <../part1-jetstream2/accessing-jetstream2>`. SUs burn for every
hour the instance exists, at a higher rate for GPU flavors, independent of
whether the annexed EP is running jobs. Shelve or delete the instance as soon
as its stage completes; an idle annex VM is the easiest way to silently drain
a cloud award.

**OSPool.** Nothing to charge: jobs that run on opportunistic capacity draw on
no allocation. If you have a PATh credit account, dedicated capacity is
charged to it in the same spirit as an HPC allocation.

Credentials follow the same split. Each resource authenticates you the normal
way when you provision the annex — your ACCESS ID (plus Duo) for Anvil's Open
OnDemand or SSH, your ACCESS credentials in Exosphere for Jetstream2 — and the
execute points authenticate back to your access point using the token material
that ``htcondor annex create`` bundles into the setup tarball. No resource
needs credentials for any other resource; the AP is the only place that sees
them all.

.. list-table::
   :header-rows: 1
   :widths: 14 24 30 32

   * - Stage
     - Runs on
     - Allocation charged
     - Where the project ID appears
   * - ``preprocess``
     - Jetstream2 annex
     - ACCESS award (SUs)
     - allocation selected at instance launch
   * - ``train``
     - Anvil annex (GPU)
     - ACCESS or NAIRR award (GPU-hours)
     - ``#SBATCH --account`` in ``hpc.slurm``
   * - ``infer``
     - OSPool
     - none (opportunistic)
     - n/a — fair-share
   * - orchestration (AP)
     - OSPool access point
     - none
     - n/a

That table is the whole funding model of the tutorial's pipeline: one free
control plane, allocations spent only on the two stages that need dedicated
capacity, and each annex tagged with the project ID that pays for it. Swap in
your own Explore award or NAIRR grant, and the same DAG runs unchanged.
