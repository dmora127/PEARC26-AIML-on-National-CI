Exploring the Dataset
=====================

The BirdCLEF training data isn't a tidy benchmark corpus. It's a large pile of
crowd-sourced field recordings — contributed to `xeno-canto
<https://xeno-canto.org/>`_ and iNaturalist by thousands of different people,
using different microphones, in different places, over a span of decades — sorted
into one directory per species and accompanied by a ``train.csv`` carrying the
labels, taxonomy, and per-recording metadata. On your Jetstream2 instance it
lives on the volume you attached in :doc:`accessing-jetstream2`:

.. code-block:: text

    /media/volume/birdclef-2026/
    ├── train_audio/     # .ogg recordings, one directory per species label
    └── train.csv        # labels, taxonomy, and per-recording metadata

It's tempting to skip straight to training a model. Resist that, because nearly
every decision that determines how well the model ends up working is a decision
about the *data*: what sample rate to standardize on, how long a clip to feed the
network, whether rare species need reweighting, and — most importantly — what
counts as noise and should be removed. You can't make those calls without first
seeing what you have.

Some of the most damaging problems are also invisible in the metadata. A
recording labeled with one species may have three others calling in the
background. Another may open with thirty seconds of the recordist announcing the
species, the date, and the location before a single bird is heard. Nothing in
``train.csv`` will tell you that. You find it by looking at the audio and
listening to it, which is exactly what this section is for.

That interactive, iterative style of work is what the cloud VM is good at. You
have the data sitting on a local volume, a full Python environment you control
(set up in :doc:`environment-setup`), and no queue between you and the next
question you want to ask. The batch systems in :doc:`Part 2 <../part2-anvil/index>`
and :doc:`Part 3 <../part3-ospool/index>` are the right tools for running the
same operation over the whole corpus; they're the wrong tools for poking at it.

Exploratory Analysis
--------------------

Open ``birdclef-data-exploration.ipynb`` in Jupyter and work through it. The
notebook combines plots with inline audio playback so you can see and hear the
same recording side by side. Five things are worth your attention:

**Class balance.** Count the recordings per species. Field recordings follow the
observers rather than the birds, so common, conspicuous, easy-to-reach species
accumulate thousands of clips while rare, nocturnal, or remote ones may have a
handful. Expect a long tail. How steep it is tells you whether you'll need class
weighting, resampling, or augmentation to keep the model from simply predicting
the majority species.

**Distribution of audio lengths.** Recordings run from a few seconds to many
minutes, and the shape of that distribution drives two choices: the clip length
you'll standardize on, and how many fixed-length windows the corpus will yield
once you slice it up. That second number is your real training-set size — it's
usually far larger than the file count.

**Spectrogram visualizations.** Plot a few recordings as spectrograms to see the
data the way the model will. A bird call shows up as a bright, localized arc or a
stack of harmonics; wind and rain show up as broadband smears across the whole
frequency range. Learning to tell those apart by eye makes the later
preprocessing decisions much less abstract.

**Audio playback.** This is the check no plot replaces, and the one most often
skipped. Listen to a handful of clips from different species. Most BirdCLEF
training clips are *focal* recordings — someone pointing a microphone at one
bird — but "focal" doesn't mean clean: you'll hear background species, insects,
traffic, wind, and handling noise, and the ``primary_label`` names only the bird
the recordist was aiming at. You'll also hear the narration.

**Data-quality issues.** Look for clips that fail to decode, files that are
near-silent end to end, unexpected sample rates or channel counts, duplicates,
and rows in ``train.csv`` with no matching audio file. It's much cheaper to find
these now than to have one of them crash a preprocessing run forty minutes into
the full corpus.

.. tip::
   Work on a sample first. Reading every file's duration means decoding every
   file, which is slow enough to break your concentration. Draw a few hundred
   recordings at random, get the shape of the answer, and only scale up if
   something looks suspicious.

Deciding What to Preprocess
---------------------------

Exploration isn't an end in itself — the point is to leave this section with a
list of things to fix. Each observation above maps onto a candidate
preprocessing step:

- **Resample to a common rate.** Contributors' equipment varies, and the model
  needs one consistent rate across every input. This tutorial standardizes on
  32 kHz, BirdCLEF's native rate, and keeps it through the entire pipeline.
- **Normalize loudness.** Recording distance and microphone gain vary enormously
  between contributors. Without normalization, a model can learn to recognize
  recording conditions instead of species.
- **Trim or pad to a fixed length.** A convolutional classifier needs
  fixed-shape input, so variable-length recordings have to become uniform
  windows one way or another.
- **Drop silent or empty segments.** The long gaps between calls carry the
  recording's label but none of its signal, and training on them teaches the
  model that silence means whatever species happened to be labeled.
- **Remove human speech and other unwanted noise.** The narration you heard
  during playback is noise for a bird classifier — and unlike the others, it's
  actively misleading, since it correlates perfectly with the very label you're
  trying to predict.

That last item turns out to be the big one, and it's where the next section
picks up: :doc:`preprocessing` turns these observations into a repeatable
pipeline that detects human speech in every recording and carves it out, leaving
clean bird audio ready for feature extraction.

.. note::
   The notebook is for *deciding*, not for *doing*. It's a fine place to try a
   normalization scheme on ten files and listen to the result, but the moment
   you're happy with a step it belongs in a script you can re-run, shard, and
   point at the whole corpus. That separation — explore interactively, execute
   reproducibly — is the pattern the rest of this tutorial follows.
