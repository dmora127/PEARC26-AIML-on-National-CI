Staging the Model for Inference
===============================

Training leaves you with a single checkpoint file that holds everything needed
to run the classifier again. Before moving on, this page looks at what that file
contains and why it's already portable, then walks through getting it onto the
OSPool access point so it's ready for large-scale inference in
:doc:`Part 3 <../part3-ospool/index>`.

The Checkpoint File
-------------------

There's no separate "export" step to run — the training script already produced
a self-contained artifact. Each time validation macro-F1 improved, it wrote
``best_<arch>.pt`` to your ``--out_dir`` (so the run from the previous page left
``outputs/best_efficientnet_b0.pt`` behind). A ``.pt`` file is just a
``torch.save`` archive; ours bundles the weights *and* the metadata inference
needs to interpret them:

``model_state_dict``
    The fine-tuned weights and biases — the actual result of training.
``arch``
    The torchvision architecture the weights belong to (e.g.
    ``efficientnet_b0``), so the model can be rebuilt before the weights are
    loaded.
``labels`` / ``label_to_idx`` / ``idx_to_label``
    The class list and the integer-to-species mapping, so a raw prediction like
    ``37`` can be decoded back into a ``primary_label``.
``image_size``
    The input resolution the model was trained on, so inference preprocesses
    spectrograms the same way.
``valid_f1``
    The validation macro-F1 at the moment the checkpoint was written — handy
    for confirming you're shipping the best run.

Because the file carries the architecture name and the label mapping alongside
the weights, it's fully portable: you can copy it to any machine with PyTorch
installed, rebuild the model, and run inference without needing the training
metadata or the original code paths. You can confirm what's inside without a GPU:

.. code-block:: python

    import torch

    ckpt = torch.load("outputs/best_efficientnet_b0.pt", map_location="cpu")
    print(ckpt.keys())
    print(ckpt["arch"], "| image_size:", ckpt["image_size"],
          "| classes:", len(ckpt["labels"]),
          "| valid_f1:", round(ckpt["valid_f1"], 4))

In :doc:`Part 3 <../part3-ospool/index>` the inference script reads exactly these
keys — it rebuilds the architecture from ``arch``, calls
``model.load_state_dict(ckpt["model_state_dict"])``, and uses ``idx_to_label`` to
turn predictions back into species names.

Transferring to the OSPool
--------------------------

Now that we have our trained model weights and biases (in the form of our
``best_efficientnet_b0.pt`` file), we need to get them to the system we'll be
using for our inference steps — the Open Science Pool (OSPool). There are a few
ways to go about this.

Sharing our Model via Pelican and the Open Science Data Federation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Open Science Data Federation <https://osg-htc.org/services/osdf.html>`_
(OSDF) is a shared network of data origins and caches that spans the OSG
infrastructure, and `Pelican <http://pelicanplatform.org/>`_ is the software and
protocol that moves data across it. Together they let a job name a dataset by a
federation-wide URL and pull it directly, with nearby caches serving repeat
requests so popular files don't have to be re-fetched from the origin every time.

This matters here because of *how* the model reaches each job. Rather than
copying the checkpoint onto the OSPool access point and then shipping it out with
every job, you can leave the file in place and let each job fetch it from the
OSDF straight into its sandbox at run time. When a directory under
``/anvil/projects/`` is exposed through a Pelican *origin*, the checkpoint
already sitting in your project space becomes addressable by an ``osdf://`` URL
that any OSPool job can request — no pre-staging on the submit node, and the
caches absorb the load when hundreds of jobs ask for the same model at once.

.. todo:: DO NOT REMOVE THIS: Need to confirm that Purdue Pelican Origin will expose user data space

Transferring using ``rsync`` or ``scp``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you'd rather move the file yourself, copy it straight from Anvil to your data
directory on the AP40 access point. Both ``rsync`` and ``scp`` work over SSH;
``rsync`` is the safer default since it can resume an interrupted transfer and
re-copies only what changed:

.. code-block:: shell

    # From Anvil, push the checkpoint to your OSPool AP40 data directory:
    rsync -avP outputs/best_efficientnet_b0.pt \
        <username>@ap40.uw.osg-htc.org:/ospool/ap40/data/<username>/

For a single file, ``scp`` is just as good:

.. code-block:: shell

    scp outputs/best_efficientnet_b0.pt \
        <username>@ap40.uw.osg-htc.org:/ospool/ap40/data/<username>/

Replace ``<username>`` with your OSG username in both the hostname and the
destination path. Once the transfer finishes, log into the access point and
confirm the file arrived where you expect:

.. code-block:: shell

    ssh <username>@ap40.uw.osg-htc.org
    ls -lh /ospool/ap40/data/<username>/best_efficientnet_b0.pt

With the checkpoint staged on AP40, you're ready to package it with the inference
code and submit jobs across the pool in :doc:`Part 3 <../part3-ospool/index>`.
