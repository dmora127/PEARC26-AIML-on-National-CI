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

.. todo:: Minimal submit file; ``condor_submit`` and verify one job works.

Resource Requests
-----------------

.. todo:: CPU/memory/disk requests and any GPU/OS requirements.
