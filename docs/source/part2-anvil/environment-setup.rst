Setting Up the Training Environment
===================================

We'll build this environment together, live and interactively, rather than
handing you a script and asking you to trust it. Anvil doesn't give you the
``sudo`` you had on Jetstream2 (:doc:`../part1-jetstream2/environment-setup`) —
a shared HPC cluster instead offers a curated stack of software **modules** plus
whatever you install into your own user-level Python environment, and getting
those two layers to agree is most of the work. Doing it by hand in a live
session means you see each step's output as it happens: which module versions
the cluster actually provides, whether the PyTorch build you pull matches the
CUDA runtime on the GPU nodes, and what a broken import looks like while there's
still a prompt in front of you to fix it at. That last point is the real
motivation for going slowly here. From :doc:`training-job` onward everything
runs unattended through Slurm, where a missing package doesn't announce itself
until your job has cleared the queue and started — so it's worth spending a few
interactive minutes now to avoid losing an hour to a one-line traceback later.
