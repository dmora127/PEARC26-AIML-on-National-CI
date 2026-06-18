Scaling Out the Inference
=========================

.. todo::

   Turn the single job into many: shard the inputs and use ``queue`` to launch
   one job per shard. Then monitor and troubleshoot the batch at scale.

Sharding the Workload
---------------------

.. todo:: Split inputs into shards; map shards to jobs.

Queueing Many Jobs
------------------

.. todo:: ``queue`` over a list/files; submit the full batch.

Monitoring & Troubleshooting
----------------------------

.. todo:: ``condor_q``, held jobs, log inspection, and retries.
