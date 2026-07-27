Activity: Deconstructing Workflows for the National CI
======================================================

Parts 1–3 handed you a finished answer: preprocess on Jetstream2, train on
Anvil, run inference on the OSPool. That mapping was chosen for you, and it was
chosen because each stage had a shape that matched one paradigm — interactive
and customizable, tightly coupled and GPU-bound, or embarrassingly parallel.
This activity is where you make that call yourself.

Working in small groups, you'll take a workflow apart of a provided example and break it
into discrete stages. For each stage, decide where it belongs and be ready to
defend the choice:

- **Which paradigm fits, and why?** Cloud, HPC, or HTC, argued from the shape of
  the work rather than from familiarity with a particular system.
- **What has to move between stages?** How much data, in which direction, and by
  what mechanism.
- **What depends on what?** Which stages must finish before others can start,
  and which could just as well run at the same time.

The seams usually turn out to be the hard part. Choosing a resource for a single
stage is reasonably straightforward once you know the shape of the work;
deciding how a large intermediate dataset gets from a cloud VM onto an HPC
scratch filesystem, or what should happen when one job out of a thousand fails,
is where real workflows get complicated — and where an orchestration layer earns
its keep.

Groups will share their plans afterward. Expect disagreement: the same stage can
reasonably land on different resources depending on data size, allocation, and
deadline, and comparing those trade-offs is more useful than converging on a
single right answer.