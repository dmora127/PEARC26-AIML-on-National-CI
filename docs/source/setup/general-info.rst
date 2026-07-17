General Information & Prerequisites
===================================

Complete these steps **before** the hands-on portion of the tutorial so you can
follow along live.

.. contents::
   :local:
   :depth: 1

Accounts & Allocations
----------------------

The tutorial spans three national CI systems — Jetstream2, Purdue Anvil, and
the OSPool — but you do **not** need to request your own allocations on any of
them. The organizers have set up a shared tutorial allocation, and completing
the single registration below provisions everything you need:

- An **OSPool account** on the ``ap40`` access point, used for the
  high-throughput inference portion (Part 3).
- Membership in the tutorial's **ACCESS-CI allocation**, which grants access to
  Jetstream2 (Part 1) and Anvil (Part 2) through your ACCESS-CI ID.

To setup your account, you will need:

#. An **institutional identity** at a US-based institution that is part of the
   InCommon federation (most US universities and labs are). If your institution
   is not in InCommon, you can register with a Google or GitHub account, but
   this requires additional manual verification — contact the tutorial
   organizers for assistance.
#. An **ACCESS-CI ID**. If you don't have one, register at
   https://access-ci.org/register *before* starting the enrollment below —
   the enrollment form asks for it. If you already have one, confirm it is
   active by signing in at https://access-ci.org with your institutional
   identity.

The full step-by-step registration walkthrough — with screenshots of the
enrollment form, email verification, and the account information email — is on
the :doc:`account-registration` page. Complete it at least **24 hours before
the tutorial** so account provisioning has time to finish.

Getting Help
------------

For assistance or questions, please email the support team at
support@osg-htc.org.

Local Tools
-----------

The tutorial is designed to run almost entirely through your **web browser** —
Jetstream2 is managed through the Exosphere web dashboard, Anvil offers Open
OnDemand and Jupyter interfaces, the OSPool offers Jupyter interfaces to the AP
and the guide itself embeds every script you need. We **highly recommend**
using these web interfaces to avoid local installation issues.

- **A modern web browser** (required). We recommend Google Chrome or Firefox
  whenever possible.
- **An SSH client** (optional, but useful). macOS and Linux include ``ssh`` in
  the terminal; on Windows, use the built-in OpenSSH client in PowerShell or
  `PuTTY <https://www.putty.org/>`_. SSH is the primary way to reach the OSPool
  access point (``ssh <username>@ap40.uw.osg-htc.org``) and an alternative way
  to reach Anvil.
- **A code editor with remote support** (optional). If you prefer editing files
  locally, `VS Code <https://code.visualstudio.com/>`_ with the Remote-SSH
  extension works well with all three systems — but the web-based editors and
  notebooks are sufficient for everything in this tutorial.

No local Python installation is needed: all computation happens on the remote
systems.

Getting the Tutorial Materials
------------------------------

There is nothing to download ahead of time:

- **Code** — every script used in the tutorial (preprocessing, training,
  inference, and the job/submit files) is embedded directly in this guide in
  collapsible code blocks, with a copy button in the upper-right corner of each
  block. The relevant page of each part tells you where to place the file on
  that system.
- **Data** — the BirdCLEF audio dataset and intermediate products (cleaned
  audio, spectrograms, trained model checkpoints) are pre-staged on the shared
  tutorial storage of each system, so each part can be started independently.
  The paths are given on the staging page of each part.

Pre-Tutorial Checklist
----------------------

Run through this list before the session; everything above the line should be
done at least **24 hours before the tutorial** so provisioning has time to
complete.

#. ☐ I have an active **ACCESS-CI ID** and can sign in at
   https://access-ci.org with my institutional identity.
#. ☐ I completed the **enrollment form** at the
   `new user enrollment page <https://osg-htc.org/pearc26-ml-enroll>`_
   (or linked my existing OSPool account to my ACCESS-CI ID).
#. ☐ I **verified my email address** via the link from registry@cilogon.org.
#. ☐ I received the **account information email** and saved my OSPool
   username.
#. ☐ I have a **modern web browser** (Chrome or Firefox recommended).
#. ☐ *(Optional)* I have an SSH client and can open a terminal.

If any of the account steps fail, email support@osg-htc.org or contact the
tutorial organizers — don't wait until the session starts.
