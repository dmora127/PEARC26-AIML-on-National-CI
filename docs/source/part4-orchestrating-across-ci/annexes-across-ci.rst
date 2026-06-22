Running Parts 1–3 with HTCondor Annexes
=======================================

.. todo::

   Show how an HTCondor Annex borrows capacity from resources like Jetstream2
   and Anvil and folds it into the pool you submit to, so the Part 1–3 stages
   can run from a single workflow within the OSPool ecosystem.

What Is an HTCondor Annex?
--------------------------

.. todo:: Concept: provisioning external resources as execute nodes that join your pool.

Annexing Cloud Capacity (Jetstream2)
------------------------------------

.. todo:: Stand up an annex on Jetstream2 to run the Part 1 preprocessing stage.

Annexing HPC Capacity (Anvil)
-----------------------------

.. todo:: Request an annex on Anvil (via Slurm) to run the Part 2 GPU training stage.

Tying the Stages Together
-------------------------

.. todo:: Route each stage to the right annex and hand results to the OSPool inference stage.
