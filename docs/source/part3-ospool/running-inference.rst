Running Inference on Soundscapes
================================

In :doc:`Part 2 <../part2-anvil/training-job>` you trained an EfficientNet-B0
classifier on Anvil and staged the result — a single checkpoint,
``best_efficientnet_b0.pt``, that carries the model weights along with the
label mapping needed to turn its outputs back into species names. Training was
the *expensive, one-time* step: it needed a GPU, a fixed block of reserved
time, and the whole labelled dataset in one place. That's exactly the workload
HPC is built for.

Now the shape of the problem flips. We have a *trained* model and a large pile
of new, unlabelled audio to run it against, where every recording is
independent of every other. Nothing has to talk to anything else, no GPU is
required, and the work grows simply by adding more recordings. As covered in
:doc:`accessing-ospool`, that is the textbook case for **high-throughput
computing**: instead of one big machine, we spread thousands of small,
self-contained inference jobs across the OSPool and let them run wherever slots
free up.

The Data: El Silencio Soundscapes
---------------------------------

The audio we want to classify is a set of **soundscapes** recorded at the El
Silencio preserve — long, passive field recordings captured by microphones left
running in the environment. Each one is a **1-minute** clip, and unlike the
training data it is genuinely *polyphonic*: many animals from many directions,
calling over and around each other, plus wind, rain, insects, and the
occasional human voice.

That's a real shift from what the model saw during training. The BirdCLEF
training clips were largely **focal recordings** — someone pointing a
microphone at *one* bird — so each labelled example centred on a single source.
A soundscape has no such focus: at any instant several species may be audible at
once, and long stretches may contain nothing of interest. The model, though,
only knows how to answer one narrow question: *given a single 5-second,
256×256 mel-spectrogram, which species is this?* Bridging that gap — from a
1-minute polyphonic recording to inputs the model recognizes — is what the
inference pipeline does.

One Soundscape, Many Clips
--------------------------

The key idea is to break each soundscape back down into the **same kind of input
the model was trained on**, classify each piece, and then reassemble the results
into a per-soundscape list of detections. Concretely, one inference job takes a
single 1-minute recording and runs it through five steps:

#. **Transfer the recording in.** The full 1-minute clip is delivered to the
   execute node as one of the job's inputs. Because there is no shared
   filesystem on the pool, the job has to carry its audio with it — how that
   transfer is set up is covered in :doc:`packaging`.
#. **Remove human speech with Silero-VAD.** Soundscapes can pick up field
   researchers or passers-by talking, and human speech is noise for a bird
   classifier. We reuse the exact same
   `Silero-VAD <https://github.com/snakers4/silero-vad>`_ step from
   :doc:`Part 1 <../part1-jetstream2/preprocessing>` to detect and cut those
   spans, so the audio is cleaned the same way the *training* audio was.
#. **Split into 5-second subclips.** The cleaned recording is sliced into
   consecutive, non-overlapping 5-second windows — up to **twelve** from a full
   minute (fewer once any speech is removed). Each window becomes one
   independent thing to classify, and its position in the recording (e.g.
   *25–30 s*) is kept so detections can be timestamped.
#. **Generate a mel-spectrogram per clip.** Each 5-second clip is converted to a
   log-scaled mel-spectrogram using the **identical settings from Part 1** — 32
   kHz audio, 256 mel bands × 256 frames, ``fmin`` 50 Hz, ``fmax`` 14 kHz (see
   :doc:`../part1-jetstream2/generating-mels`). Matching those settings is not
   optional: the model was trained on inputs of that exact shape and frequency
   range, so anything else would feed it images it never learned to read.
#. **Classify each clip.** Every spectrogram is passed through the loaded
   ``best_efficientnet_b0.pt`` model, which produces a probability across the
   known species. The label mapping stored in the checkpoint turns the winning
   index (or every index above a confidence threshold) back into species names.

The output is a **list of detected calls** for that one soundscape — which
species were heard, and in which 5-second windows — written to a small results
file. Multiplied across all the soundscapes and all their windows, that's the
large-scale classification we set out to do.

.. note::

   Because each 5-second clip is scored on its own, a single window can surface
   *more than one* species when several are calling at once — the classifier
   isn't forced to pick just one answer per soundscape. That's what lets a
   polyphonic recording yield a genuine list of detections rather than a single
   label.

Why This Fits the OSPool
------------------------

Look at the pipeline above and notice what's *absent*: no step needs any other
soundscape, and no clip needs any other clip. Every recording can be processed
in complete isolation. That independence is precisely the property that makes
this a *massively parallel* problem and a natural fit for the OSPool:

- **One job per soundscape** (or per small batch of them) means the work splits
  cleanly into hundreds of jobs that run concurrently across the pool, so the
  whole collection finishes in roughly the time of a single recording rather
  than one-after-another.
- **Each job is modest and short** — a few minutes of CPU on one recording — so
  if the opportunistic resources reclaim a slot mid-run, only that one job is
  lost and it simply reruns elsewhere.
- **No GPU required.** Inference on a single 5-second spectrogram is cheap, so
  jobs run on ordinary CPU cores, which the pool has in vast supply.

.. tip::

   The whole pipeline — Silero-VAD, the mel-spectrogram generation, and the
   model — is reused almost verbatim from Parts 1 and 2. The one genuinely new
   requirement is making all of it run *anywhere* on the pool: the model, the
   code, and every dependency have to travel with the job. That portability is
   the subject of the next page.

With the inference workload clear, the next step is to package the model and
this pipeline into a self-contained job that runs identically on any machine in
the pool — the subject of :doc:`packaging`.
