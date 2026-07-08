Accessing Anvil
===============

The mel-spectrograms you staged at the end of
:doc:`Part 1 <../part1-jetstream2/staging-data>` are now sitting in Anvil's
project space, so the next step is to log into Anvil itself.
`Anvil <https://www.rcac.purdue.edu/anvil>`_ is Purdue University's ACCESS HPC
cluster: roughly a thousand CPU nodes plus GPU nodes carrying four NVIDIA A100s
apiece, all scheduled with Slurm — the batch system you'll meet in
:doc:`training-job`. This page covers the two ways to log in, a tour of the
filesystems, and where your data, code, and job outputs should live.

Logging In
----------

Anvil offers two ways in, and both land you in the same account, filesystems,
and resources:

- **Open OnDemand** — a web portal with a file browser, job composer, and
  in-browser terminal. Nothing to install or configure; all you need is a
  browser and your ACCESS credentials.
- **SSH** — the traditional command-line route, which requires a registered
  SSH key.

If you're unsure, use Open OnDemand: it's the fastest way in, and everything in
this tutorial can be done from its built-in shell.

Via Open OnDemand
^^^^^^^^^^^^^^^^^

1. Navigate to the Anvil Open OnDemand portal at
   https://ondemand.anvil.rcac.purdue.edu/ and sign in with your ACCESS-CI
   credentials.

   .. image:: ../assets/purdue/access-login.png
      :alt: ACCESS-CI single sign-on page
      :width: 80%
      :align: center

2. Complete two-factor authentication if prompted.

   .. image:: ../assets/purdue/ACCESS-Duo.png
      :alt: ACCESS-CI Duo two-factor authentication page
      :width: 80%
      :align: center

3. From the dashboard you can reach file management, job submission, and
   interactive apps. For now, open **Clusters** and select **>_ Anvil Shell
   Access** to start a terminal session on an Anvil login node.

   .. image:: ../assets/purdue/OOD-splash.png
      :alt: Purdue Anvil Open OnDemand dashboard
      :width: 80%
      :align: center

4. You now have a shell on Anvil — from here you can navigate the filesystems,
   manage files, and submit jobs to the cluster.

   .. image:: ../assets/purdue/anvil-term.png
      :alt: Anvil terminal session in Open OnDemand
      :width: 80%
      :align: center

Via SSH
^^^^^^^

Anvil accepts standard SSH connections with public-key authentication to
``anvil.rcac.purdue.edu``, using your *Anvil* username:

.. code-block:: shell

    ssh x-accessusername@anvil.rcac.purdue.edu

.. note::

   - Your Anvil username is **not** the same as your ACCESS username, although
     it is derived from it: Anvil usernames start with an ``x-`` prefix, e.g.
     ``x-accessusername``.
   - Password authentication is not supported. There is no "Anvil password",
     and your ACCESS password will not be accepted by Anvil's SSH either.
     Instead, register an SSH public key, which you can do from the Open
     OnDemand interface — follow the `SSH keys instructions in the Anvil user
     guide <https://docs.rcac.purdue.edu/userguides/anvil/getting-started/#ssh-keys>`_.
   - When reporting SSH problems to the help desk, run the ``ssh`` command with
     ``-vvv`` and include the verbose output in your ticket.

The Filesystems
---------------

Every Anvil node mounts the same storage areas, each reachable through an
environment variable and each with its own quota, backup guarantee, and purge
policy. Three of them matter for this tutorial:

.. list-table::
   :header-rows: 1
   :widths: 14 22 14 12 20 18

   * - Area
     - Location
     - Quota
     - Snapshots
     - Purge policy
     - Use it for
   * - ``$HOME``
     - ``/home/<username>``
     - 25 GB
     - Yes [#snap]_
     - Not purged
     - Code, scripts, small software installs
   * - ``$PROJECT``
     - ``/anvil/projects/<allocation>``
     - 5 TB per allocation
     - Yes [#snap]_
     - Kept while the allocation is active; removed 90 days after it expires
     - Data shared across the allocation — datasets, common software
   * - ``$SCRATCH``
     - ``/anvil/scratch/<username>``
     - 100 TB / 1M files
     - No
     - Files unaccessed for 30 days are purged
     - High-performance job I/O — checkpoints, temporary outputs

.. [#snap] Nightly snapshots are kept for 7 days, weekly snapshots for 3
   weeks, and monthly snapshots for 2 months.

Everyone on the same allocation shares read and write access to the
``$PROJECT`` space, which makes it the natural home for shared inputs — it's
where the tutorial's preprocessed dataset landed at the end of Part 1.
(``$WORK`` also exists and points to the same location as ``$PROJECT``; the
two are interchangeable.)

``$SCRATCH`` lives on a 10 PB GPFS parallel filesystem that can deliver up to
150 GB/s, which is why job I/O belongs there rather than in your home
directory.

.. warning::

   Scratch is fast but *transient*: it is not backed up, and files are purged
   once they haven't been accessed for 30 days. If a file is deleted, lost to
   a disk failure, or purged, it cannot be restored — copy anything you care
   about (like a finished model checkpoint) to project space or off the
   cluster.

To check your usage and quotas on each filesystem, run ``myquota``:

.. code-block:: text

    x-accessusername@login03.anvil:[~] $ myquota

    Type     Location          Size       Limit      Use     Files    Limit    Use
    ==============================================================================
    home     x-accessusername   261.5MB    25.0GB     1%       -       -        -
    scratch  anvil             6.3GB      100.0TB    0.01%    3k      1,048k   0.36%
    projects accountname1      37.2GB     5.0TB      0.73%    403k    1,048k   39%
    projects accountname2      135.8GB    5.0TB      3%       20k     1,048k   2%

For Part 2, the layout looks like this: the mel-spectrograms staged in Part 1
sit in the allocation's ``$PROJECT`` space, where every participant's training
jobs can read them; your copy of the training script and job files lives in
``$HOME``; and training outputs — logs and model checkpoints — are written to
``$SCRATCH``, with the best checkpoint copied somewhere durable before it's
staged for the OSPool in :doc:`staging-model`.

Transferring Files
------------------

The heavy lifting is already done — the dataset was moved onto Anvil with
``rsync`` at the end of Part 1 (see
:doc:`../part1-jetstream2/staging-data`). For the smaller transfers you'll
still make, Anvil supports the usual options:

- **scp / rsync** — both work from any Linux or macOS machine (and from
  Windows SSH clients) against ``anvil.rcac.purdue.edu``, using the same
  ``x-`` username and SSH key as your login. For example, to pull a file from
  Anvil to your laptop:

  .. code-block:: shell

      scp x-accessusername@anvil.rcac.purdue.edu:/anvil/scratch/x-accessusername/outputs/best_model.pt .

- **Open OnDemand** — the portal's **Files** app lets you browse, upload, and
  download files from your home, scratch, and project directories directly in
  the browser, which is often the quickest route for a single file.

- **Globus** — a web-based transfer and sharing service that connects Anvil to
  other ACCESS sites, campus systems, and personal machines, with transfers
  that are managed for you and survive interruptions. It's the right tool for
  large or long-running transfers; see `Using Globus with ACCESS resources
  <https://access-ci.atlassian.net/wiki/spaces/ACCESSdocumentation/pages/552861697/Using+Globus>`_.

With a shell on Anvil and a clear picture of where files belong, the next step
is building the software environment the training job will run in:
:doc:`environment-setup`.
