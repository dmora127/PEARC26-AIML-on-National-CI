The Orchestration Model
=======================

So far each part of this tutorial has treated a single resource as a
destination: you logged into Jetstream2 to preprocess, into Anvil to train, and
into an OSPool access point to run inference. Part 4 reframes that picture.
Instead of three places you *go*, think of HTCondor and the OSPool/PATh as a
single **control plane** — one layer you submit work to, which then places that
work onto whatever capacity is available across the national CI.

The shift is from "log in and run a job here" to "describe the work once and let
the system decide where it runs." The Part 1–3 stages don't change; what changes
is that preprocess → train → infer become one coordinated workflow with a single
point of submission, rather than three disconnected sessions you stitch together
by hand.

.. admonition:: What you'll take away
   :class: tip

   - The access-point model: submit once, let HTCondor place work.
   - HTCondor Annexes as a way to grow *your* pool with borrowed cloud and HPC
     capacity.
   - Expressing the pipeline's stage dependencies as a workflow (DAGMan).
   - Where PATh and the OSPool fit as the control plane over heterogeneous
     resources.

Submit Locally, Run Everywhere
------------------------------

The foundation of this model is the HTCondor **access point** (AP). You prepare
your jobs in one place — submit files, input data, and a description of the
resources each job needs — and hand them to the access point with
``condor_submit``. From that moment, you are no longer responsible for *where*
the work runs.

HTCondor matches each job against execute points (EPs) that can satisfy its
requirements and dispatches the work there, transferring inputs out and results
back automatically. A job that asks for a GPU lands on a machine with a GPU; a
job that asks for a particular OS or amount of memory lands somewhere that
matches. The access point handles queuing, retries, and file movement, so the
same submission behaves the same way whether it runs on one pool or several.

.. todo::

   Walk through a minimal end-to-end submission from an OSPool access point:
   ``condor_submit`` → ``condor_q`` → results returned. Reuse the Part 3 job as
   the concrete example so the access-point view connects back to known work.

The important consequence: the *set* of machines your jobs can reach is not
fixed. By default it's whatever the pool offers, but you can expand it — which
is what Annexes are for.

Expanding *Your* Compute Capacity with HTCondor Annexes
-------------------------------------------------------

By default, your jobs run on whatever the pool happens to offer. But you often
hold allocations on resources the pool *doesn't* already include — a Jetstream2
instance, an Anvil GPU allocation, even an idle lab workstation. An **HTCondor
Annex** lets you put that capacity to work without leaving the access-point
model.

The idea is simple: an annex provisions machines from a resource you have access
to and turns them into HTCondor execute points that report back to your access
point. From then on they're just more places your jobs can land. You don't
change how you submit, and you don't log in to babysit the borrowed nodes — you
keep queuing work, and HTCondor matches it onto the annexed capacity alongside
everything else.

That's what makes "submit once, run everywhere" real for *your* allocations
specifically:

- **Use the allocations you already have.** Fold NAIRR, ACCESS, and campus
  resources into the same pool instead of running each one by hand.
- **Right-size each stage.** Send GPU training to an Anvil annex, interactive-
  style preprocessing to a Jetstream2 annex, and high-throughput inference to
  the OSPool — all from one workflow.
- **Grow and shrink on demand.** Stand up capacity when a stage needs it and let
  it go when the stage is done.

The companion page, :doc:`annexes-across-ci`, walks through the mechanics —
standing up annexes on Jetstream2 and Anvil and routing the Part 1–3 stages onto
them.

Expressing the Pipeline as a Workflow
-------------------------------------

Placing individual jobs is only half the story. The Part 1–3 stages have an
*order*: inference can't start until training produces a model, and training
can't start until preprocessing produces its inputs. Submitting each stage by
hand and watching for it to finish before kicking off the next defeats the point
of a control plane.

**DAGMan** (Directed Acyclic Graph Manager) lets you declare those dependencies
once and hand the whole thing to the access point. You describe each stage as a
node and the edges between them, and DAGMan submits each node only after its
parents complete successfully — retrying, holding, or aborting per your rules.
From your side it's a single submission: DAGMan runs as a job itself, walks the
graph, and submits each stage at the right time.

The running example as a DAG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tutorial's bird-call classifier is a clean three-node chain. Each node is an
ordinary HTCondor submit file — the same ones you'd run by hand — and the edges
encode the artifact hand-off between stages:

.. list-table::
   :header-rows: 1
   :widths: 16 24 30 30

   * - Node
     - Runs on
     - Consumes
     - Produces
   * - ``preprocess``
     - Jetstream2 annex
     - raw recordings
     - spectrogram PNGs + ``training_metadata.csv``
   * - ``train``
     - Anvil annex (GPU)
     - spectrograms + metadata
     - ``best_efficientnet_b0.pt`` (weights + label map)
   * - ``infer``
     - OSPool (sharded)
     - the checkpoint + input shards
     - per-shard prediction files

.. code-block:: text

   # pipeline.dag — preprocess -> train -> infer
   JOB  preprocess  preprocess.sub
   JOB  train       train.sub
   JOB  infer       infer.sub

   PARENT preprocess CHILD train
   PARENT train      CHILD infer

   # Re-run a flaky stage a few times before giving up on the whole pipeline.
   RETRY train 2
   RETRY infer 3

.. code-block:: console

   $ condor_submit_dag pipeline.dag

DAGMan now owns the pipeline: ``train`` is not submitted until ``preprocess``
reports success, and the sharded ``infer`` batch is not submitted until the
checkpoint exists. If a stage fails, DAGMan retries it up to its ``RETRY`` count
and, if it still fails, halts and writes a *rescue DAG* so you can fix the
problem and resume from the failed node instead of rerunning the whole chain.

Routing each stage to the right resource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing in ``pipeline.dag`` says *where* a stage runs — that lives in each node's
submit file. The ``preprocess`` and ``train`` nodes target the annexes you stood
up earlier (so they land on Jetstream2 and Anvil respectively), while ``infer``
takes the pool's default OSPool capacity. The GPU and OS requirements that each
stage already declares do the rest of the routing for you.

.. todo::

   Pin down and show the exact submit-file syntax for targeting a named annex
   from a DAG node (e.g. the ``--annex-name`` equivalent as a submit command),
   versus letting ``infer`` match default OSPool EPs. Verify against the running
   HTCondor version.

Passing artifacts between stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The edges in the DAG enforce *order*, but the actual bytes — spectrograms, the
checkpoint, prediction shards — still have to travel between resources that don't
share a filesystem. Two DAGMan hooks handle that glue:

- A **PRE script** runs before a node's job is submitted. Use it to stage that
  stage's inputs into place — e.g. shard the inference input list, or fetch the
  checkpoint to where ``infer.sub`` expects it.
- A **POST script** runs after a node's job finishes. Use it to publish that
  stage's outputs to shared storage and to check success beyond the exit code.

.. code-block:: text

   # Stage train's checkpoint out to shared storage, then shard inputs for infer.
   SCRIPT POST train  stage_out_model.sh   best_efficientnet_b0.pt
   SCRIPT PRE  infer  make_shards.sh       inputs/ shards/

Where those artifacts actually live between stages — OSDF/Pelican origins,
Globus collections, or annex-local scratch — and how each job fetches them is
the subject of :doc:`moving-data-and-models`; the DAG just decides *when* each
transfer happens.

Submitting and watching the pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ condor_submit_dag pipeline.dag
   $ condor_q                      # the DAGMan job plus whatever stage is live
   $ tail -f pipeline.dag.dagman.out   # human-readable progress through the graph

.. todo::

   Add a small left-to-right diagram of the three nodes with the artifact passed
   on each edge, mirroring the table above.

PATh and the OSPool as a Control Plane
--------------------------------------

The access point, annexes, and DAGMan together turn the OSPool/PATh ecosystem
into a control plane rather than a single venue. You submit to one place; the
pool abstracts over a heterogeneous mix of resources — OSPool execute points,
cloud instances on Jetstream2, HPC allocations on Anvil — and presents them as
one schedulable surface.

.. todo::

   - Clarify the OSPool vs. PATh distinction and which access point a tutorial
     participant actually uses.
   - Explain how job requirements (GPU, OS, memory) route work to the right
     subset of a mixed pool.
   - Tie back to the Part 4 goals: one workflow, many resources, no rewrite per
     system.
