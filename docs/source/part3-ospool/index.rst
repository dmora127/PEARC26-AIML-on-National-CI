Part 3 — Large-Scale Inference on the OSPool
============================================

In the final part, you'll take the model trained in
:doc:`Part 2 <../part2-anvil/index>` and run high-throughput batch inference
across the OSPool, distributing the work over many independent HTCondor jobs.

.. admonition:: Goals for this section
   :class: tip

   - Log into an OSPool access point and understand the HTC model.
   - Package the model and inference code so jobs are self-contained.
   - Write an HTCondor submit file and split the workload across many jobs.
   - Submit, monitor, and troubleshoot a large batch of inference jobs.
   - Collect and aggregate the results.

.. admonition:: Why HTC for inference?
   :class: note

   Batch inference is *embarrassingly parallel* — each input is independent —
   which makes it a natural fit for high-throughput computing rather than a
   single large machine.

.. toctree::
   :maxdepth: 1

   accessing-ospool
   packaging
   submit-file
   scaling-out
   collecting-results
