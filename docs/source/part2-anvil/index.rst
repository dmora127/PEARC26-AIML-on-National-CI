Part 2 — Model Training on Purdue Anvil
=======================================

With the preprocessed data from :doc:`Part 1 <../part1-jetstream2/index>`, you'll
now train the model on Purdue Anvil, a national HPC cluster. You'll move from
interactive prototyping to GPU-accelerated batch jobs scheduled with Slurm.

.. admonition:: Goals for this section
   :class: tip

   - Log into Anvil and understand the cluster layout.
   - Set up the training environment and access the staged data.
   - Request GPU resources and submit a training job with Slurm.
   - Monitor the job and checkpoint the trained model.
   - Stage the trained model for inference on the OSPool.

.. toctree::
   :maxdepth: 1
   :hidden:

   accessing-anvil
   environment-setup
   training-job
   monitoring
   staging-model
