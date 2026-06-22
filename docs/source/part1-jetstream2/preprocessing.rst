Building the Preprocessing Pipeline
===================================

The exploration in the previous section showed that the raw BirdCLEF
recordings aren't ready to train on as-is: many clips are crowd-sourced field
recordings (think `xeno-canto <https://xeno-canto.org/>`_ and iNaturalist),
and a sizeable fraction contain the recordist narrating — announcing the
species, the location, or the date. That human speech is noise for a bird
classifier, so the goal of this section is to turn the ad-hoc exploration into
a **repeatable pipeline** that cleans the audio and produces consistent inputs
for model training on Anvil.

The pipeline runs in two stages, each a small command-line tool you run on your
Jetstream2 instance:

#. **Detect** where human speech occurs in every recording (Silero-VAD).
#. **Remove** that speech, keeping only the clean bird audio (librosa +
   soundfile).

The cleaned audio is then ready for feature extraction (mel-spectrograms) and
transfer to Anvil, which we cover in :doc:`staging-data`.

The Pipeline
------------

Rather than one monolithic script, the pipeline is split into discrete stages
that each read an artifact and write the next one — a file list, then a CSV of
speech regions, then the cleaned audio. This design is deliberate:

- **Restartable** — if a stage is interrupted, you re-run it without redoing
  the work before it. The CSV from the detection stage is the durable handoff
  to the removal stage.
- **Re-runnable** — you can re-run detection with a different threshold, or
  re-run removal in a different output format, without re-crawling the
  filesystem or recomputing earlier stages.
- **Shardable** — because each stage works from an explicit file list, the work
  splits cleanly across many independent jobs. That's exactly the property
  we'll exploit later when fanning the same kind of work out across the OSPool
  in :doc:`Part 3 <../part3-ospool/index>`.

Each tool takes a ``--workers`` flag so you can use all of the cores on your
Jetstream2 instance; parallelism comes from processing many files at once
rather than many threads per file.

.. tip::
   Start with ``--limit`` (a handful of files) to confirm the command runs and
   the output looks right before launching it over the full corpus.

Detecting Human Speech in the Audio Files using Silero-VAD
----------------------------------------------------------

The first stage finds *where* humans are talking in each recording.
`Silero-VAD <https://github.com/snakers4/silero-vad>`_ is a lightweight,
pre-trained voice-activity-detection model: given an audio stream, it returns
the time spans that contain human speech. We run it over every ``.ogg`` file
and record those spans — we don't modify any audio yet.

The tool works in two steps so you crawl the directory tree once and can then
re-run detection as many times as you like:

#. **crawl** walks the dataset and writes the list of ``.ogg`` files (a fast,
   single-threaded pass over the filesystem).
#. **vad** runs Silero-VAD across that list — parallelized across files with
   ``--workers`` — and writes one CSV row per recording.

The example below crawls and runs detection in one shot:

.. code-block:: shell

    python3 preprocessing.py vad --input /media/volume/birdclef-2026/train_audio/ -o speech_regions.csv --workers 8

The result is ``speech_regions.csv``, with one row per recording recording its
duration, the number of speech segments, the total speech seconds and ratio,
and the detected ``[start, end]`` spans (in seconds). Errors are captured
per-row rather than aborting the run, so a single unreadable file won't sink
the whole pass.

.. note::
   Silero-VAD is trained on human speech, so on dense nature audio it can
   occasionally fire on non-speech sounds. Treat ``--threshold`` as a knob —
   higher is stricter — and spot-check a few rows before trusting the run. This
   is also why detection only *records* speech regions; nothing is removed
   until the next stage, where you can review the spans first.

.. collapse:: Code: Detect Human Speech with Silero-VAD

    .. code-block:: python

        #!/usr/bin/env python3
        """
        Detect human-speech regions in .ogg recordings with Silero-VAD.

        Two steps:
          1. crawl  -- walk a directory tree and write the list of .ogg files
          2. vad    -- run Silero voice-activity detection over that list (parallel
                       across files via --workers), writing a CSV with one row per
                       file listing the [start, end] spans (seconds) flagged as speech.

        Splitting the two means you crawl the tree once, then re-run VAD with different
        thresholds -- or shard the file list across OSPool jobs -- without re-walking
        the filesystem.

        Intended for cleaning bioacoustic corpora (e.g. xeno-canto / iNaturalist
        clips) where recordist narration contaminates the audio. Note: Silero is
        trained on human speech, so on dense nature audio it can occasionally fire
        on non-speech sounds -- treat --threshold as a knob and spot-check results.

        Dependencies:
            pip install silero-vad librosa soundfile tqdm    # torch is pulled in by silero-vad

        Usage:
            # 1. find the files (fast, single-threaded crawl)
            python detect_speech.py crawl ./train_audio -o ogg_files.txt

            # 2. run VAD over them, N files at a time
            python detect_speech.py vad --file-list ogg_files.txt -o speech_regions.csv --workers 8

            # ...or run it in one shot -- crawls, writes the manifest, then runs VAD:
            python detect_speech.py vad --input ./train_audio --manifest ogg_files.txt --workers 8
        """

        from __future__ import annotations

        import argparse
        import csv
        import json
        import multiprocessing as mp
        import os
        import sys
        import time
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from pathlib import Path

        import librosa
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        # speech_segments holds a JSON list of [start_sec, end_sec] pairs.
        CSV_FIELDS = [
            "relpath",
            "filepath",
            "duration_s",
            "n_speech_segments",
            "speech_seconds",
            "speech_ratio",
            "speech_segments",
            "error",
        ]


        def find_ogg_files(root: Path):
            """Yield every .ogg file under root (recursive, case-insensitive)."""
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() == ".ogg":
                    yield p


        def read_file_list(path: Path):
            """Read a crawl manifest: one path per line; blank lines and # comments skipped."""
            files = []
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        files.append(Path(line))
            return files


        def analyze_file(path: Path, model, sampling_rate: int, device: str, vad_kwargs: dict):
            """Run VAD on one file. Returns (duration_s, [[start, end], ...] in seconds)."""
            audio, _ = librosa.load(str(path), sr=sampling_rate, mono=True)  # float32 mono
            duration_s = round(len(audio) / sampling_rate, 3)
            wav = torch.from_numpy(audio).to(device)
            stamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate, **vad_kwargs)
            segments = [
                [round(s["start"] / sampling_rate, 3), round(s["end"] / sampling_rate, 3)]
                for s in stamps
            ]
            return duration_s, segments


        def build_row(path: Path, root: Path, model, sampling_rate: int, device: str, vad_kwargs: dict):
            """Run VAD on one file and return a full CSV row dict (errors captured inline)."""
            row = {k: "" for k in CSV_FIELDS}
            row["filepath"] = str(path)
            row["relpath"] = os.path.relpath(str(path), str(root))
            try:
                duration_s, segments = analyze_file(path, model, sampling_rate, device, vad_kwargs)
                speech_seconds = round(sum(e - s for s, e in segments), 3)
                row["duration_s"] = duration_s
                row["n_speech_segments"] = len(segments)
                row["speech_seconds"] = speech_seconds
                row["speech_ratio"] = round(speech_seconds / duration_s, 4) if duration_s else 0.0
                row["speech_segments"] = json.dumps(segments)
            except Exception as exc:  # one bad file shouldn't kill the crawl
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["speech_segments"] = json.dumps([])
            return row


        # --- worker-process plumbing (used only when --workers > 1) ---
        # Each worker loads its own model once via the pool initializer and pins torch
        # to a single thread, so parallelism comes from running many files at once
        # rather than many threads per file.
        _WORKER: dict = {}


        def _init_worker(root: Path, sampling_rate: int, device: str, vad_kwargs: dict):
            torch.set_num_threads(1)
            model = load_silero_vad()
            model.to(device)
            _WORKER.update(root=root, model=model, sampling_rate=sampling_rate,
                           device=device, vad_kwargs=vad_kwargs)


        def _process_path(path_str: str):
            w = _WORKER
            return build_row(Path(path_str), w["root"], w["model"],
                             w["sampling_rate"], w["device"], w["vad_kwargs"])


        def run_vad(files, relroot: Path, args):
            """Run VAD across `files`, writing args.output. Sequential or pooled by args.workers."""
            vad_kwargs = dict(
                threshold=args.threshold,
                min_speech_duration_ms=args.min_speech_ms,
                min_silence_duration_ms=args.min_silence_ms,
                speech_pad_ms=args.speech_pad_ms,
            )

            n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
            n_workers = min(n_workers, len(files))
            print(f"Processing {len(files)} file(s) with {n_workers} worker(s)"
                  + (" (sequential)" if n_workers == 1 else "")
                  + f"; relpath base = {relroot}")

            def rows_sequential():
                model = load_silero_vad()
                model.to(args.device)
                for p in files:
                    yield build_row(p, relroot, model, args.sampling_rate, args.device, vad_kwargs)

            def rows_parallel():
                ctx = mp.get_context("spawn")  # safest start method alongside torch
                with ProcessPoolExecutor(
                    max_workers=n_workers, mp_context=ctx,
                    initializer=_init_worker,
                    initargs=(relroot, args.sampling_rate, args.device, vad_kwargs),
                ) as ex:
                    futures = [ex.submit(_process_path, str(p)) for p in files]
                    for fut in as_completed(futures):
                        yield fut.result()

            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(files), unit="file", desc="VAD")
            except ImportError:
                pbar = None
                print("(tqdm not installed -- printing periodic progress instead)", file=sys.stderr)

            args.output.parent.mkdir(parents=True, exist_ok=True)
            n_with_speech = n_errors = 0
            t0 = time.time()

            rows = rows_parallel() if n_workers > 1 else rows_sequential()
            with open(args.output, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for i, row in enumerate(rows, 1):
                    writer.writerow(row)
                    fh.flush()

                    if row["error"]:
                        n_errors += 1
                    elif row["n_speech_segments"]:
                        n_with_speech += 1

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(speech=n_with_speech, errors=n_errors)
                    elif i % 50 == 0 or i == len(files):
                        print(f"  {i}/{len(files)} files "
                              f"({n_with_speech} with speech, {n_errors} errors)", file=sys.stderr)

            if pbar is not None:
                pbar.close()

            dt = time.time() - t0
            print(
                f"Done in {dt:.1f}s -> {args.output}\n"
                f"  files processed : {len(files)}\n"
                f"  with speech     : {n_with_speech}\n"
                f"  errors          : {n_errors}"
            )


        def crawl_to_manifest(root: Path, output: Path):
            """Walk `root` for .ogg files, write them one-per-line to `output`, return (root, files)."""
            root = root.resolve()
            if not root.is_dir():
                sys.exit(f"error: not a directory: {root}")
            files = sorted(find_ogg_files(root))
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as fh:
                fh.writelines(f"{p}\n" for p in files)
            print(f"Found {len(files)} .ogg file(s) under {root}")
            print(f"Wrote file list -> {output}")
            return root, files


        def cmd_crawl(args):
            crawl_to_manifest(args.input, args.output)


        def cmd_vad(args):
            if args.file_list:
                # use the pre-built manifest as-is
                files = read_file_list(args.file_list)
                source = f"file list {args.file_list}"
                crawl_root = None
            else:
                # no --file-list: crawl first, write the manifest, then run VAD on it
                crawl_root, files = crawl_to_manifest(args.input, args.manifest)
                source = f"directory {crawl_root}"

            if args.limit:
                files = files[: args.limit]
            if not files:
                print(f"No .ogg files to process (from {source})", file=sys.stderr)
                return

            if args.root:
                relroot = args.root.resolve()
            elif crawl_root is not None:
                relroot = crawl_root  # known true root -> relpaths stay stable
            elif len(files) > 1:
                relroot = Path(os.path.commonpath([str(p) for p in files]))
            else:
                relroot = files[0].parent

            run_vad(files, relroot, args)


        def main():
            parser = argparse.ArgumentParser(
                description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
            )
            sub = parser.add_subparsers(dest="command", required=True)

            pc = sub.add_parser("crawl", help="Find all .ogg files under a directory and write the list")
            pc.add_argument("input", type=Path, help="Root directory to crawl")
            pc.add_argument("-o", "--output", type=Path, default=Path("ogg_files.txt"),
                            help="Where to write the .ogg path list, one per line (default ogg_files.txt)")
            pc.set_defaults(func=cmd_crawl)

            pv = sub.add_parser("vad", help="Run Silero-VAD across a file list (or a directory)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            grp = pv.add_mutually_exclusive_group(required=True)
            grp.add_argument("--file-list", type=Path, help="Path list produced by `crawl` (one .ogg per line)")
            grp.add_argument("--input", type=Path, help="Directory to crawl inline instead of using a file list")
            pv.add_argument("-o", "--output", type=Path, default=Path("speech_regions.csv"),
                            help="Output CSV path")
            pv.add_argument("--manifest", type=Path, default=Path("ogg_files.txt"),
                            help="Where to write the file list when crawling via --input (ignored with --file-list)")
            pv.add_argument("--root", type=Path, default=None,
                            help="Base dir for the relpath column (default: common ancestor of the files)")
            pv.add_argument("--workers", type=int, default=1,
                            help="Worker processes for parallel files (1 = sequential; <=0 = all cores)")
            pv.add_argument("--threshold", type=float, default=0.5,
                            help="Speech probability threshold; higher = stricter")
            pv.add_argument("--min-speech-ms", type=int, default=250,
                            help="Discard speech chunks shorter than this")
            pv.add_argument("--min-silence-ms", type=int, default=100,
                            help="Silence needed to end a chunk")
            pv.add_argument("--speech-pad-ms", type=int, default=30,
                            help="Pad each detected chunk on both sides")
            pv.add_argument("--sampling-rate", type=int, default=16000, choices=[8000, 16000],
                            help="Rate the VAD runs at; audio is resampled to this")
            pv.add_argument("--device", default="cpu", help="Torch device for the model")
            pv.add_argument("--limit", type=int, default=None,
                            help="Process at most N files (handy for a dry run)")
            pv.set_defaults(func=cmd_vad)

            args = parser.parse_args()
            args.func(args)


        if __name__ == "__main__":
            main()

|

Removing Human Speech from the Audio Files using Librosa
--------------------------------------------------------

The second stage uses the CSV from detection to actually clean the audio. For
each recording it **inverts** the speech spans — everything *not* flagged as
speech is the clean bird audio — slices those non-speech spans out, and
concatenates them into a new file. It optionally also writes the carved-out
speech to a separate directory, which is handy for spot-checking what the VAD
caught.

A short total of detected speech (below ``--min-speech-seconds``) is treated as
a false positive, and the clip is kept whole as clean audio. The ``--only-carved``
flag goes a step further: for clips that were never touched, it records them in
the CSV but doesn't rewrite them, since the original ``.ogg`` already *is* the
clean audio — avoiding needless duplication of most of the corpus.

.. code-block:: shell

    python3 ./speech_separator.py  --workers 16 --speech-dir /media/volume/birdclef-working-dir/speech_audio/ --non-speech-dir /media/volume/birdclef-working-dir/non_speech_audio/ --format wav

This writes the cleaned audio and extends the CSV with two columns:
``non_speech_segments`` (the clean spans) and ``clean_audio_path`` (the file to
feed downstream). That ``clean_audio_path`` column is the contract with the
feature-extraction step: it points at the clean audio for every recording,
whether that's a newly carved file or an untouched original.

.. tip::
   Audio I/O here uses librosa and soundfile rather than torchaudio, so the
   stage doesn't depend on torchaudio's ``sox_effects`` (removed in 2.9).
   ``--format flac`` is lossless and much smaller than ``wav``; timestamps are
   stored in seconds, so the saved sample rate (``--rate``, default 32 kHz —
   BirdCLEF's native rate) can change without invalidating the CSV.

.. collapse:: Code: Separate Speech/Non-Speech with Librosa + Soundfile

    .. code-block:: python

        #!/usr/bin/env python3
        """
        Separate speech from non-speech audio using the VAD output (speech_regions.csv).

        For EVERY .ogg row in the input CSV:
          * compute the non-speech (clean) spans by inverting the speech spans,
          * write the clean audio to --non-speech-dir,
          * and, only when meaningful speech was detected, write the carved-out speech
            to --speech-dir.

        Files with no detected speech -- or only trivial speech below MIN_SPEECH_SECONDS
        -- are treated as entirely clean. By default their whole clip is written as the
        non-speech output; with --only-carved they are recorded in the CSV but NOT
        re-materialized (the original .ogg already is the clean audio), which avoids
        duplicating most of the corpus.

        The output CSV gains two columns:
          * non_speech_segments -- the clean [start, end] spans (seconds)
          * clean_audio_path    -- the audio to feed downstream (make_mels.py): the
                                   written _non_speech file, or the original .ogg when
                                   kept whole under --only-carved, or "" if the clip was
                                   entirely speech / failed.

        Audio IO uses librosa (read) + soundfile (write) -- no silero/torchaudio, so it
        won't break when torchaudio drops sox_effects in 2.9. Saved at --rate (default
        32 kHz, BirdCLEF's native rate); CSV timestamps are in seconds so they map to
        any rate. --format flac is lossless and much smaller than wav.

        Dependencies:
            pip install librosa soundfile pandas tqdm

        Usage:
            python separate_speech.py --workers 16 \
                --speech-dir /media/volume/birdclef-2026/speech_audio \
                --non-speech-dir /media/volume/birdclef-2026/non_speech_audio \
                --format flac
        """

        import argparse
        import ast
        import json
        import multiprocessing as mp
        import os
        import sys
        from concurrent.futures import ProcessPoolExecutor, as_completed

        import numpy as np
        import pandas as pd
        import librosa
        import soundfile as sf


        MIN_SPEECH_SECONDS = 0.999  # speech below this -> file treated as entirely clean
        EXT = {"wav": ".wav", "flac": ".flac"}


        def parse_segments(x):
            if pd.isna(x) or x == "" or x == "[]":
                return []
            if isinstance(x, list):
                return x
            try:
                return json.loads(x)
            except Exception:
                return ast.literal_eval(x)


        def clean_segments(segments, duration_s):
            cleaned = []
            for seg in segments:
                if len(seg) != 2:
                    continue
                start, end = float(seg[0]), float(seg[1])
                start = max(0.0, min(start, duration_s))
                end = max(0.0, min(end, duration_s))
                if end > start:
                    cleaned.append([start, end])
            cleaned.sort(key=lambda x: x[0])
            return cleaned


        def total_segment_seconds(segments):
            return sum(end - start for start, end in segments)


        def invert_segments(speech_segments, duration_s, min_segment_s=0.0):
            non_speech = []
            prev_end = 0.0
            speech_segments = clean_segments(speech_segments, duration_s)
            for start, end in speech_segments:
                if start > prev_end:
                    gap = [prev_end, start]
                    if gap[1] - gap[0] >= min_segment_s:
                        non_speech.append(gap)
                prev_end = max(prev_end, end)
            if prev_end < duration_s:
                gap = [prev_end, duration_s]
                if gap[1] - gap[0] >= min_segment_s:
                    non_speech.append(gap)
            return non_speech


        def cut_and_concat(audio, segments_s, sr):
            """Slice [start, end]-second spans from a mono numpy array and concatenate them."""
            n = len(audio)
            pieces = []
            for start, end in segments_s:
                a = max(0, int(round(start * sr)))
                b = min(n, int(round(end * sr)))
                if b > a:
                    pieces.append(audio[a:b])
            if not pieces:
                return np.zeros(0, dtype=audio.dtype)
            return np.concatenate(pieces)


        def _write(out_path, audio, sr):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, audio, sr)


        def process_record(i, rec, cfg):
            """Cut one file. Returns (index, non_speech_json, clean_audio_path, status)."""
            filepath = rec["filepath"]
            result = "[]"
            clean_audio_path = ""
            try:
                duration_s = float(rec["duration_s"])
                speech_segments = clean_segments(parse_segments(rec["speech_segments"]), duration_s)

                ss = rec.get("speech_seconds")
                speech_seconds = float(ss) if (ss is not None and not pd.isna(ss)) else total_segment_seconds(speech_segments)

                has_speech = len(speech_segments) > 0 and speech_seconds >= cfg["min_speech"]
                if not has_speech:
                    speech_segments = []

                non_speech_segments = invert_segments(speech_segments, duration_s)

                # With --only-carved, a kept-whole (no-speech) file is recorded but not rewritten.
                write_clean = bool(non_speech_segments) and (not cfg["only_carved"] or bool(speech_segments))

                if speech_segments or write_clean:
                    audio, _ = librosa.load(filepath, sr=cfg["rate"], mono=True)
                    base = os.path.splitext(rec["relpath"])[0]
                    if speech_segments:
                        sp_chunk = cut_and_concat(audio, speech_segments, cfg["rate"])
                        if len(sp_chunk):
                            _write(os.path.join(cfg["speech_dir"], base + "_speech" + cfg["ext"]), sp_chunk, cfg["rate"])
                    if write_clean:
                        ns_out = os.path.join(cfg["non_speech_dir"], base + "_non_speech" + cfg["ext"])
                        ns_chunk = cut_and_concat(audio, non_speech_segments, cfg["rate"])
                        if len(ns_chunk):
                            _write(ns_out, ns_chunk, cfg["rate"])
                            clean_audio_path = ns_out

                # Kept-whole under --only-carved: the original file already is the clean audio.
                if not clean_audio_path and non_speech_segments and cfg["only_carved"] and not speech_segments:
                    clean_audio_path = filepath

                result = json.dumps(non_speech_segments)
                status = "carved" if speech_segments else "clean"
            except Exception as e:
                status = f"failed: {type(e).__name__}: {e}"
            return i, result, clean_audio_path, status


        # --- worker plumbing (used only when --workers > 1) ---
        _CFG: dict = {}


        def _init_worker(cfg):
            _CFG.update(cfg)


        def _pool_worker(payload):
            i, rec = payload
            return process_record(i, rec, _CFG)


        def process_all(records, cfg, n_workers):
            results = {}
            counts = {"carved": 0, "clean": 0, "failed": 0}

            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(records), unit="file", desc="Separating speech/non-speech")
            except ImportError:
                pbar = None

            def tally(i, res, clean_path, status):
                results[i] = (res, clean_path)
                if status.startswith("failed"):
                    counts["failed"] += 1
                    msg = f"Failed on {records[i]['filepath']}: {status[8:]}"
                    (tqdm.write(msg) if pbar is not None else print(msg, file=sys.stderr))
                else:
                    counts[status] += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(carved=counts["carved"], clean=counts["clean"], failed=counts["failed"])

            if n_workers == 1:
                for i, rec in enumerate(records):
                    tally(*process_record(i, rec, cfg))
            else:
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                         initializer=_init_worker, initargs=(cfg,)) as ex:
                    futures = [ex.submit(_pool_worker, (i, rec)) for i, rec in enumerate(records)]
                    for fut in as_completed(futures):
                        tally(*fut.result())

            if pbar is not None:
                pbar.close()

            non_speech_col = [results.get(i, ("[]", ""))[0] for i in range(len(records))]
            clean_path_col = [results.get(i, ("[]", ""))[1] for i in range(len(records))]
            return non_speech_col, clean_path_col, counts


        def main():
            ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
            ap.add_argument("--input-csv", default="speech_regions.csv", help="VAD output CSV")
            ap.add_argument("--output-csv", default="speech_regions_with_nonspeech.csv",
                            help="CSV to write (adds non_speech_segments and clean_audio_path columns)")
            ap.add_argument("--speech-dir", default="speech_audio", help="Where carved speech audio goes")
            ap.add_argument("--non-speech-dir", default="non_speech_audio", help="Where clean audio goes")
            ap.add_argument("--rate", type=int, default=32000, help="Sample rate for saved audio")
            ap.add_argument("--format", choices=["wav", "flac"], default="wav",
                            help="Output format; flac is lossless and much smaller")
            ap.add_argument("--min-speech-seconds", type=float, default=MIN_SPEECH_SECONDS,
                            help="Total speech below this -> file kept whole as clean")
            ap.add_argument("--only-carved", action="store_true",
                            help="Only write clean audio for files that had speech removed "
                                 "(don't re-materialize untouched clips)")
            ap.add_argument("--workers", type=int, default=1,
                            help="Worker processes (1 = sequential; <=0 = all cores)")
            args = ap.parse_args()

            df = pd.read_csv(args.input_csv)
            records = df.to_dict("records")

            n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
            n_workers = min(n_workers, max(1, len(records)))

            cfg = dict(
                rate=args.rate,
                ext=EXT[args.format],
                min_speech=args.min_speech_seconds,
                speech_dir=args.speech_dir,
                non_speech_dir=args.non_speech_dir,
                only_carved=args.only_carved,
            )

            print(f"Processing {len(records)} row(s) with {n_workers} worker(s); "
                  f"format={args.format}, rate={args.rate}, only_carved={args.only_carved}")

            non_speech_col, clean_path_col, counts = process_all(records, cfg, n_workers)

            df["non_speech_segments"] = non_speech_col
            df["clean_audio_path"] = clean_path_col
            df.to_csv(args.output_csv, index=False)

            n_clean_paths = sum(1 for p in clean_path_col if p)
            print(f"Rows processed            : {len(df)} "
                  f"({counts['carved']} carved, {counts['clean']} kept whole, {counts['failed']} failed)")
            print(f"Rows with clean audio     : {n_clean_paths}")
            print(f"Saved updated CSV to      : {args.output_csv}")
            print(f"Saved speech audio to     : {args.speech_dir}")
            print(f"Saved non-speech audio to : {args.non_speech_dir}")


        if __name__ == "__main__":
            main()

|
