Scaling Out the Inference
=========================

The previous page ended with one job that works: a single soundscape, cleaned,
sliced, classified, and returned. Turning that into the whole collection is a
surprisingly small change, and the reason is the property we identified back in
:doc:`running-inference` — no soundscape needs any other soundscape. Because the
work was already independent, scaling out doesn't require new machinery for
splitting or coordinating it. You tell HTCondor to build one job per recording
from a list, and it handles matching those jobs to machines across the pool.

What *does* change is how you supervise the run. With one job you can read its
log line by line; with several hundred you stop watching individual jobs and
start watching the population — how many are idle, how many are running, and
which handful got stuck and need attention. This page covers both halves:
queueing many jobs from a single submit file, and keeping track of them once
they're out there.

Queueing Many Jobs
------------------

HTCondor's ``queue <var> from <file>`` syntax reads a text file and creates one
job per line, binding each line to a variable you can reference elsewhere in the
submit file. Start by listing the soundscapes you want to process, one per line,
in ``soundscapes.txt``:

.. code-block:: text

    ES_01_20230115_060000.ogg
    ES_01_20230115_061000.ogg
    ES_02_20230115_060000.ogg

Then point the ``queue`` statement at it:

.. code-block:: text

    queue soundscape from soundscapes.txt

That single line replaces the bare ``queue`` from the single-job submit file. If
``soundscapes.txt`` has 500 lines, ``condor_submit`` creates 500 jobs, each with
``$(soundscape)`` set to its own line. Everywhere else in the submit file,
``$(soundscape)`` now refers to *this* job's recording — which is what lets one
description produce hundreds of distinct jobs:

.. code-block:: text

    transfer_input_files = osdf:///osg-public/pearc26-aiml/data/$(soundscape), best_efficientnet_b0.pt

Each job now transfers its own soundscape plus the shared model checkpoint to
whatever execute node it lands on.

.. important::
   Give every job its own output and log filenames, or they will overwrite each
   other as results come back to the same directory. HTCondor provides
   ``$(Cluster)`` (the submission's ID) and ``$(Process)`` (this job's index
   within it) for exactly this:

   .. code-block:: text

       output = logs/infer_$(Cluster)_$(Process).out
       error  = logs/infer_$(Cluster)_$(Process).err
       log    = logs/infer_$(Cluster).log

   A shared ``log`` file is fine and often preferable — HTCondor writes one
   event per job to it, so it becomes a single record of the whole run. It's
   ``output`` and ``error`` that must be unique per job.

Each line can just as easily name a small batch rather than a single recording.
If your soundscapes are short enough that per-job overhead — transferring the
container and the model — starts to rival the actual work, group ten or twenty
per line and have the job loop over them. That trade-off (fewer, longer jobs
versus more, shorter ones) is worth tuning once you've seen how long a single
job actually takes.

Monitoring & Troubleshooting
----------------------------

Checking the Queue
^^^^^^^^^^^^^^^^^^

``condor_q`` shows your jobs. By default it summarizes a submission as a single
batch, which is what you want when there are hundreds:

.. code-block:: shell

    condor_q

.. code-block:: text

    OWNER     BATCH_NAME     SUBMITTED   DONE   RUN    IDLE   HOLD  TOTAL  JOB_IDS
    jane.doe  ID: 4521      7/26 13:58    312     47     139      2    500  4521.0-499

Read it left to right: 312 finished, 47 running right now, 139 still waiting for
a match, and 2 held. Idle counts are normal and not a problem — the OSPool is a
shared, opportunistic system, and jobs wait for slots to free up. The column to
care about is ``HOLD``, which means HTCondor stopped those jobs and is waiting
for you.

To see individual jobs instead of the summary, add ``-nobatch``:

.. code-block:: shell

    condor_q -nobatch

The ``ST`` column gives each job's state: ``I`` for idle, ``R`` for running,
``H`` for held.

Held Jobs
^^^^^^^^^

A hold is HTCondor's way of saying "this job cannot proceed and re-running it
unchanged won't help." Ask why:

.. code-block:: shell

    condor_q -hold

The hold reason is usually specific and actionable. The common ones on the
OSPool are a file transfer failure (a soundscape named in ``soundscapes.txt``
that doesn't exist at that path, or a typo in an ``osdf://`` URL) and a job
exceeding the memory or disk it requested. Fix the underlying cause, then put
the jobs back in the queue:

.. code-block:: shell

    condor_release <cluster_id>          # or a single job: condor_release 4521.17

If the fix requires editing the submit file, remove the held jobs with
``condor_rm`` and resubmit instead — ``condor_release`` re-runs them with their
original description.

Jobs that stay idle indefinitely are a different problem: nothing in the pool
matches what you asked for. ``condor_q -better-analyze <job_id>`` reports which
of your requirements is rejecting machines, which is typically an over-large
memory or disk request narrowing the pool to almost nothing.

