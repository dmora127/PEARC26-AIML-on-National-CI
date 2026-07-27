Monitoring & Checkpointing
==========================

Submitting with ``sbatch`` (:doc:`training-job`) hands your job to Slurm — and
hands away the immediate feedback you had on Jetstream2. There's no console to
watch and no traceback in front of you when something breaks: a job may sit in
the queue for minutes or hours before it starts, and once it runs, everything it
has to say goes to files rather than to your terminal. This page is about
following a job from queued to finished — finding where it sits in the queue,
reading its logs while it runs, and confirming that the checkpoint the training
script writes each time validation macro-F1 improves actually landed on disk
before you move on to :doc:`staging-model`.

Monitoring the Job
------------------

``sbatch`` printed a job ID when you submitted, and that number is the handle
for everything below. Slurm answers two different questions with two different
commands: ``squeue`` shows where a *queued or running* job stands right now,
while ``sacct`` reports what happened to a job — including after it finishes.
For what the training run itself is doing, you'll read the log file it writes
and check the GPU it's using.

Queue Status with ``squeue``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    squeue -u $USER

.. code-block:: text

    JOBID     PARTITION  NAME      ST  TIME   NODES  NODELIST(REASON)
    1234567   gpu        birdclef  R   12:34  1      g006

Read the ``ST`` (state) column first. ``PD`` means pending — the job is waiting
in line, and ``NODELIST(REASON)`` shows why: ``(Priority)`` (other jobs are
ahead of yours) and ``(Resources)`` (waiting for a GPU to free up) are both
normal, and neither needs any action from you. Once the state flips to ``R``
the job is running, ``TIME`` counts elapsed walltime, and the node name (here
``g006``) appears in the last column. Note that ``squeue`` only shows jobs that
are still in the system — when your job finishes (or dies), it vanishes from
this list entirely rather than showing a "done" state.

If you spot a mistake — wrong data path, too-short walltime — you don't have to
let the job run its course:

.. code-block:: shell

    scancel <job_id>

Job History with ``sacct``
^^^^^^^^^^^^^^^^^^^^^^^^^^

``sacct`` queries Slurm's accounting database, so it works on finished jobs —
it's the tool for "my job disappeared from ``squeue``; how did it go?". The
default output is sparse, so ask for the useful columns:

.. code-block:: shell

    sacct -j <job_id> --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,ExitCode

.. code-block:: text

    JobID           JobName    Partition  State      Elapsed   MaxRSS  ExitCode
    --------------- ---------- ---------- ---------- -------- -------- --------
    1234567         birdclef+  gpu        COMPLETED  00:52:11             0:0
    1234567.batch   batch                 COMPLETED  00:52:11   14.2G      0:0
    1234567.extern  extern                COMPLETED  00:52:11             0:0

Each job shows up as several rows: the parent job plus *steps* like ``.batch``
(your script) and ``.extern``. Some fields only appear on the step rows —
``MaxRSS`` (peak memory) is reported on ``.batch``, not the parent. The states
you'll actually see:

- ``COMPLETED`` — the script exited cleanly (``ExitCode`` ``0:0``).
- ``FAILED`` — the script exited non-zero; the log file will say why.
- ``TIMEOUT`` — the job hit the ``-t`` walltime limit and Slurm killed it.
- ``OUT_OF_MEMORY`` — the job exceeded its ``--mem`` request.
- ``CANCELLED`` — someone (usually you, via ``scancel``) stopped it.

``MaxRSS`` is also your feedback loop for the batch script: if a completed run
peaked at 14 GB, the ``--mem=20G`` request was about right; if it peaked at
4 GB, you can request less next time and likely start sooner.

Following the Training Log
^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything the training script prints lands in ``slurm-<job_id>.out`` in the
directory you submitted from, and the file grows as the job runs. ``tail -f``
follows it live:

.. code-block:: shell

    tail -f slurm-<job_id>.out

.. code-block:: text

    Using device: cuda
    Training examples: 8231
    Validation examples: 2058
    Classes: 84
    Epoch 001 | train_loss=2.9147 train_acc=0.3320 train_f1=0.2544 | valid_loss=2.0233 valid_acc=0.5511 valid_f1=0.4145
    Saved new best model to outputs/best_efficientnet_b0.pt with valid_f1=0.4145
    Epoch 002 | train_loss=1.9418 train_acc=0.5304 train_f1=0.4482 | valid_loss=1.4996 valid_acc=0.6417 valid_f1=0.5344
    Saved new best model to outputs/best_efficientnet_b0.pt with valid_f1=0.5344

The number to watch is ``valid_f1`` — it should climb over the first few epochs,
and each improvement is followed by a "Saved new best model" line confirming a
fresh checkpoint. Press ``Ctrl-C`` when you're done watching; that stops
``tail``, not the job.

.. tip::
   Glance at the log as soon as the job starts running. Import errors, a
   mistyped ``--metadata_csv`` path, or a missing module die within the first
   few seconds — catching that immediately beats discovering it after the job
   sat in the queue and you waited an hour for "results."

Checking GPU Utilization
^^^^^^^^^^^^^^^^^^^^^^^^

The log tells you training is *progressing*; it doesn't tell you whether the
GPU you requested is actually being kept busy. ``nvidia-smi`` reports that, but
it has to run on the compute node where the job lives. ``srun --overlap``
attaches a command to your job's existing allocation:

.. code-block:: shell

    # One snapshot:
    srun --jobid=<job_id> --overlap --pty nvidia-smi

    # Refresh every 5 seconds (Ctrl-C to stop):
    srun --jobid=<job_id> --overlap --pty nvidia-smi -l 5

Two numbers in the output matter here:

``GPU-Util``
    Should sit high (roughly 70–100%) while an epoch is running, with natural
    dips during validation and between epochs. If it hovers *low* throughout,
    the GPU is starved — usually the DataLoader can't feed it fast enough.
    Raise ``--num_workers`` (and ``--cpus-per-task`` to match) or increase
    ``--batch_size``.
``Memory-Usage``
    How much of the GPU's memory the job occupies. Lots of headroom means you
    could raise ``--batch_size`` or try a larger ``--arch``; running near the
    ceiling explains any CUDA out-of-memory crashes in the log.


