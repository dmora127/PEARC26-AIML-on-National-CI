Collecting & Aggregating Results
================================

The queue drains and the work is done — but "done" leaves you with a directory
full of small CSV files, one per job, each holding the detections for a single
soundscape. Individually they answer almost nothing. Together they're the result
of the entire run, and this page is about turning that pile of fragments into one
table you can actually analyze.

The step worth slowing down for is the one in the middle. An empty queue is not
the same as a successful run: on an opportunistic pool a job can exit non-zero,
sit held until you notice, or fail in a way that returns an empty file rather
than no file at all. None of that announces itself — a missing shard just
quietly shrinks your result set, and a summary computed over 487 of 500
soundscapes looks exactly as plausible as one computed over all 500. So the
order here is gather, *verify*, then aggregate.

Gathering Outputs
-----------------

Three different kinds of file come back from a run, and it's worth being precise
about which is which:

``predictions_<N>.csv``
    The actual results, named by ``transfer_output_files`` in the submit file.
    This is your data.
``logs/infer_<cluster>_<proc>.out`` / ``.err``
    Whatever your script wrote to stdout and stderr. Diagnostics, not results.
``logs/infer_<cluster>.log``
    HTCondor's own event record for the whole submission — one entry per job for
    submitted, executing, evicted, and terminated, with exit codes.

.. note::
   The predictions arrive through ``transfer_output_files``, *not* through the
   ``output`` file. ``output`` only captures what your script printed; if
   ``run_inference.py`` writes its CSV to disk (which it does), that file has to
   be listed for transfer or it never leaves the execute node.

Start with the cheapest possible check — did you get back as many files as you
submitted jobs?

.. code-block:: shell

    ls predictions_*.csv | wc -l
    wc -l < soundscapes.txt

If those two numbers match, you're in good shape. If they don't, ``condor_history``
tells you what happened to the jobs that are no longer in the queue. It's the
OSPool's counterpart to ``sacct`` from :doc:`Part 2 <../part2-anvil/monitoring>`:

.. code-block:: shell

    condor_history <cluster_id> -af ProcId ExitCode

Any job whose exit code isn't ``0`` failed, and its ``.err`` file will say why:

.. code-block:: shell

    condor_history <cluster_id> -af ProcId ExitCode | awk '$2 != 0'

Failures on a heterogeneous pool tend to sort into two groups. A handful of jobs
failing scattered across the run usually means something transient — a bad node,
a timed-out transfer — and simply re-running those inputs is the right fix. Every
job failing, or every job in one shard, points at something systematic in the
script, the container, or the input list, and re-running will just reproduce it.
Read one ``.err`` file before deciding which case you're in.

To re-run the stragglers, write the soundscapes for the failed jobs into a new
list file and submit it exactly as before — the submit file doesn't need to
change, only the file the ``queue`` statement reads:

.. code-block:: shell

    condor_submit run_inference.sub

Aggregating the Predictions
---------------------------

Once you have a complete set, combining it is a concatenation — every file
carries the same columns, so the merged table is just the rows stacked up. Doing
this in pandas handles the header rows for you and gives you somewhere to put
the sanity checks:

.. code-block:: python

    import glob
    import pandas as pd

    files = sorted(glob.glob("predictions_*.csv"))
    detections = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    detections.to_csv("all_predictions.csv", index=False)

    print(f"{len(files)} files -> {len(detections):,} detections")

.. tip::
   Adjust the column names in the checks below to match what your
   ``run_inference.py`` actually writes. The output described in
   :doc:`running-inference` is a list of detected calls — which species, in
   which 5-second window of which recording — so expect a recording identifier,
   a time window, a species label, and a confidence score.

Before trusting the merged file, ask it a few questions that would expose a
broken run:

- **Does every soundscape appear?** Compare the unique recording identifiers
  against ``soundscapes.txt``. A recording with zero detections is plausible;
  a recording that's entirely absent means its job's output never made it back.
- **Are the detection counts sane?** Each one-minute soundscape yields at most
  twelve 5-second windows, fewer once speech removal cuts into it. A recording
  with hundreds of detections, or the whole run with a handful, means something
  upstream went wrong.
- **Is the species distribution degenerate?** If one label dominates nearly
  every window, the most likely explanation isn't an unusually loud bird — it's
  a mismatch between the label mapping in the checkpoint and the one used at
  inference, or spectrograms generated with settings that don't match training.
- **Do the confidence scores spread out?** Scores clustered at a single value
  suggest the model isn't really discriminating between windows.

These checks are quick, and they're the difference between shipping a result and
shipping a result-shaped artifact. Once the merged table survives them, Part 3 is
complete: you've taken a model trained on one machine and applied it across a
national pool of thousands. What remains is connecting all three parts into a
single workflow rather than three manual handoffs, which is
:doc:`Part 4 <../part4-orchestrating-across-ci/index>`.
