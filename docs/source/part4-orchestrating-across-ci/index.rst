Part 4 — Orchestrating Across the National CI Ecosystem
=======================================================

In the final part, you'll step back from any single resource and treat the
national CI as one programmable system. You'll use HTCondor and the
`OSPool <https://osg-htc.org/services/open_science_pool.html>`_ / `PATh
<https://path-cc.io/>`_ as an *orchestration layer* — a control plane that can
reach into cloud, HPC, and HTC resources and run the Part 1–3 stages
(preprocess → train → infer) as a single, coordinated workflow. The key
mechanism is the **HTCondor Annex**, which lets you borrow capacity from
resources like Jetstream2 and Anvil and fold it into the pool you submit to.

.. admonition:: Goals for this section
   :class: tip

   - Understand HTCondor + OSPool/PATh as an orchestration layer over the
     national CI, not just a place to run jobs.
   - Use HTCondor Annexes to bring Jetstream2 and Anvil capacity into a single
     pool and run the Part 1–3 stages from one workflow.
   - Run on NSF ACCESS-CI and NAIRR Pilot allocations.
   - Move data and models across systems reliably.
   - Build for reproducibility and portability so the same workflow runs
     anywhere on the national CI.
   - Recognize common pitfalls, CI anti-patterns, and when *not* to reach for a
     given system.

.. admonition:: Why an orchestration layer?
   :class: note

   Parts 1–3 each used the "right tool" for one stage in isolation. Real
   workflows have to *connect* those stages — moving data, scheduling
   dependent steps, and adapting to whatever allocation is available. HTCondor
   lets you express the whole pipeline as a managed workflow and run it across
   heterogeneous resources without rewriting it for each one.

.. toctree::
   :maxdepth: 1
   :hidden:

   orchestration-model
   annexes-across-ci
   access-and-nairr-allocations
   moving-data-and-models
   reproducibility-and-portability
   pitfalls-and-anti-patterns
   when-not-to-use
   activity-deconstructing-workflows
