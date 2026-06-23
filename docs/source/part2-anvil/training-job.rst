Submitting a Training Job
=========================

With the environment in place (:doc:`environment-setup`) and the preprocessed
spectrograms staged on Anvil, you're ready to train the classifier. Anvil is a
*batch* system: rather than running training interactively, you describe the
resources you need — above all a GPU — in a Slurm batch script and submit it to
the scheduler, which queues your job and runs it on a compute node once those
resources are free. This page covers the two pieces that make that happen: the
PyTorch **training script**, and the **Slurm batch script** that requests a GPU
and launches it.

.. tip::
   While developing, it's often easier to grab an interactive GPU session first
   — on Anvil, ``sinteractive -A <account> -p gpu --gpus-per-node=1 -t 00:30:00``
   drops you onto a GPU node where you can run the script directly, watch it
   start, and shake out path or dependency problems before committing to a
   batch job. Switch to ``sbatch`` once it runs cleanly.

The Training Script
-------------------

The training script is a compact, single-GPU PyTorch image-classification loop.
Because Part 1 rendered every recording as spectrogram PNGs, we can treat
bird-call recognition as an ordinary image problem and fine-tune a model that
already knows how to see. In outline, the script:

- **Reads the training metadata** produced in Part 1, using the ``split``
  column to separate train from validation rows and mapping each
  ``primary_label`` to an integer class.
- **Loads spectrograms on demand** — ``SpectrogramDataset`` opens each PNG
  (relative to ``--mel_root``) and converts it to RGB.
- **Fine-tunes a pretrained backbone** — it pulls a torchvision model
  (EfficientNet-B0 by default) with its ImageNet weights and swaps the final
  classifier layer for one sized to the number of bird classes.
- **Augments only the training data** — time/frequency masking, frequency
  shift and edge-boost, additive noise, and MixUp make the model robust to
  variation; validation runs on clean spectrograms so its metrics stay honest.
- **Optimizes and tracks progress** — AdamW on cross-entropy loss, reporting
  loss, accuracy, and macro-F1 for both splits each epoch.
- **Checkpoints the best model** — whenever validation macro-F1 improves, it
  writes ``best_<arch>.pt`` to ``--out_dir``, bundling the weights *and* the
  label mapping so inference in :doc:`Part 3 <../part3-ospool/index>` can decode
  predictions back into species names.

.. note::
   Macro-F1 (averaged equally across classes) is the headline metric because
   the species are imbalanced — plain accuracy would flatter a model that just
   predicts the common birds. Note too that because MixUp blends training
   labels, the *train* accuracy/F1 are approximate; the validation numbers are
   the ones to trust.

A minimal run pointed at the Part 1 metadata and your staged spectrograms looks
like this:

.. code-block:: shell

    python3 train.py --metadata_csv effnet_b0_training_metadata.csv --mel_root /anvil/projects/x-cis260991/melspecs/ --out_dir outputs/ --epochs 5 --batch_size 32 --lr 1e-4 --num_workers 8 --arch efficientnet_b0

.. collapse:: Code: Model Training Script

    .. code-block:: python

        #!/usr/bin/env python3

        import argparse
        from pathlib import Path

        import pandas as pd
        import torch
        import torch.nn as nn

        from PIL import Image
        from sklearn.metrics import accuracy_score, f1_score
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        from torchvision.models import get_model, get_model_weights
        from tqdm import tqdm


        class SpectrogramDataset(Dataset):
            def __init__(self, df, label_to_idx, mel_root, transform=None):
                self.df = df.reset_index(drop=True)
                self.label_to_idx = label_to_idx
                self.mel_root = Path(mel_root)
                self.transform = transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]

                image = Image.open(self.mel_root / row["mel_path"]).convert("RGB")
                label = self.label_to_idx[row["primary_label"]]

                if self.transform is not None:
                    image = self.transform(image)

                return image, torch.tensor(label, dtype=torch.long)


        class RandomTimeMask:
            def __init__(self, max_width=48, num_masks=2, p=0.7):
                self.max_width = max_width
                self.num_masks = num_masks
                self.p = p

            def __call__(self, x):
                # x shape: [C, H, W]
                if torch.rand(1).item() > self.p:
                    return x

                x = x.clone()
                _, _, W = x.shape

                for _ in range(self.num_masks):
                    width = torch.randint(0, self.max_width + 1, (1,)).item()

                    if width == 0 or width >= W:
                        continue

                    start = torch.randint(0, W - width + 1, (1,)).item()
                    x[:, :, start:start + width] = 0.0

                return x


        class RandomFrequencyMask:
            def __init__(self, max_width=24, num_masks=2, p=0.7):
                self.max_width = max_width
                self.num_masks = num_masks
                self.p = p

            def __call__(self, x):
                # x shape: [C, H, W]
                if torch.rand(1).item() > self.p:
                    return x

                x = x.clone()
                _, H, _ = x.shape

                for _ in range(self.num_masks):
                    width = torch.randint(0, self.max_width + 1, (1,)).item()

                    if width == 0 or width >= H:
                        continue

                    start = torch.randint(0, H - width + 1, (1,)).item()
                    x[:, start:start + width, :] = 0.0

                return x


        class RandomFrequencyShift:
            def __init__(self, max_shift=8, p=0.5):
                self.max_shift = max_shift
                self.p = p

            def __call__(self, x):
                # x shape: [C, H, W]
                if torch.rand(1).item() > self.p:
                    return x

                shift = torch.randint(
                    -self.max_shift,
                    self.max_shift + 1,
                    (1,),
                ).item()

                if shift == 0:
                    return x

                x_shifted = torch.zeros_like(x)

                if shift > 0:
                    x_shifted[:, shift:, :] = x[:, :-shift, :]
                else:
                    x_shifted[:, :shift, :] = x[:, -shift:, :]

                return x_shifted


        class RandomSpectrogramNoise:
            def __init__(self, std=0.025, p=0.5):
                self.std = std
                self.p = p

            def __call__(self, x):
                # x is expected to be pre-normalization tensor in [0, 1]
                if torch.rand(1).item() > self.p:
                    return x

                noise = torch.randn_like(x) * self.std
                x = x + noise

                return torch.clamp(x, 0.0, 1.0)


        class RandomFrequencyEdgeBoost:
            def __init__(self, band_fraction=0.25, min_gain=1.0, max_gain=1.25, p=0.4):
                self.band_fraction = band_fraction
                self.min_gain = min_gain
                self.max_gain = max_gain
                self.p = p

            def __call__(self, x):
                # x shape: [C, H, W]
                # Boost either the top or bottom frequency edge.
                # This avoids assuming the spectrogram frequency orientation with certainty.
                if torch.rand(1).item() > self.p:
                    return x

                x = x.clone()
                _, H, _ = x.shape

                band_h = max(1, int(H * self.band_fraction))
                gain = self.min_gain + torch.rand(1).item() * (
                    self.max_gain - self.min_gain
                )

                if torch.rand(1).item() < 0.5:
                    x[:, :band_h, :] *= gain
                else:
                    x[:, H - band_h:, :] *= gain

                return torch.clamp(x, 0.0, 1.0)


        def mixup_data(images, labels, alpha=0.15, p=0.5):
            if alpha <= 0 or torch.rand(1).item() > p:
                return images, labels, labels, 1.0

            lam = torch.distributions.Beta(alpha, alpha).sample().item()
            batch_size = images.size(0)
            index = torch.randperm(batch_size, device=images.device)

            mixed_images = lam * images + (1.0 - lam) * images[index]
            labels_a = labels
            labels_b = labels[index]

            return mixed_images, labels_a, labels_b, lam


        def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
            return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(
                logits,
                labels_b,
            )


        def train_one_epoch(model, loader, optimizer, criterion, device):
            model.train()

            total_loss = 0.0
            all_preds = []
            all_labels = []

            for images, labels in tqdm(loader, desc="train", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                images, labels_a, labels_b, lam = mixup_data(
                    images,
                    labels,
                    alpha=0.15,
                    p=0.5,
                )

                logits = model(images)
                loss = mixup_criterion(
                    criterion,
                    logits,
                    labels_a,
                    labels_b,
                    lam,
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)

                # Note: with MixUp, train accuracy/F1 are approximate diagnostics.
                # Validation accuracy/F1 remain the main metrics.
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

            avg_loss = total_loss / len(loader.dataset)
            acc = accuracy_score(all_labels, all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

            return avg_loss, acc, macro_f1


        @torch.no_grad()
        def evaluate(model, loader, criterion, device):
            model.eval()

            total_loss = 0.0
            all_preds = []
            all_labels = []

            for images, labels in tqdm(loader, desc="valid", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                total_loss += loss.item() * images.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

            avg_loss = total_loss / len(loader.dataset)
            acc = accuracy_score(all_labels, all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

            return avg_loss, acc, macro_f1


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--metadata_csv", required=True)
            parser.add_argument("--mel_root", required=True)
            parser.add_argument("--arch", default="efficientnet_b0")
            parser.add_argument("--out_dir", default="outputs/effnet_b0_baseline")
            parser.add_argument("--epochs", type=int, default=5)
            parser.add_argument("--batch_size", type=int, default=32)
            parser.add_argument("--lr", type=float, default=1e-4)
            parser.add_argument("--num_workers", type=int, default=4)
            parser.add_argument("--image_size", type=int, default=224)
            args = parser.parse_args()

            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            df = pd.read_csv(args.metadata_csv)

            labels = sorted(df["primary_label"].unique())
            label_to_idx = {label: i for i, label in enumerate(labels)}
            idx_to_label = {i: label for label, i in label_to_idx.items()}

            train_df = df[df["split"] == "train"].copy()
            valid_df = df[df["split"] == "valid"].copy()

            print(f"Training examples: {len(train_df)}")
            print(f"Validation examples: {len(valid_df)}")
            print(f"Classes: {len(labels)}")

            train_transform = transforms.Compose(
                [
                    transforms.Resize((args.image_size, args.image_size)),

                    # Existing augmentation from the original script.
                    # For spectrograms, this is effectively time reversal.
                    transforms.RandomHorizontalFlip(p=0.5),

                    # Image-style intensity augmentation for spectrogram PNGs.
                    transforms.ColorJitter(
                        brightness=0.20,
                        contrast=0.20,
                    ),

                    transforms.ToTensor(),

                    # Spectrogram-focused augmentations before normalization.
                    RandomFrequencyShift(max_shift=8, p=0.5),
                    RandomFrequencyMask(max_width=24, num_masks=2, p=0.7),
                    RandomTimeMask(max_width=48, num_masks=2, p=0.7),
                    RandomFrequencyEdgeBoost(
                        band_fraction=0.25,
                        min_gain=1.0,
                        max_gain=1.25,
                        p=0.4,
                    ),
                    RandomSpectrogramNoise(std=0.025, p=0.5),

                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

            valid_transform = transforms.Compose(
                [
                    transforms.Resize((args.image_size, args.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

            train_dataset = SpectrogramDataset(
                train_df,
                label_to_idx=label_to_idx,
                mel_root=args.mel_root,
                transform=train_transform,
            )

            valid_dataset = SpectrogramDataset(
                valid_df,
                label_to_idx=label_to_idx,
                mel_root=args.mel_root,
                transform=valid_transform,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            weights = get_model_weights(args.arch).DEFAULT
            model = get_model(args.arch, weights=weights)

            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, len(labels))

            model = model.to(device)

            criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=1e-4,
            )

            best_f1 = -1.0

            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc, train_f1 = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )

                valid_loss, valid_acc, valid_f1 = evaluate(
                    model=model,
                    loader=valid_loader,
                    criterion=criterion,
                    device=device,
                )

                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f} "
                    f"train_f1={train_f1:.4f} | "
                    f"valid_loss={valid_loss:.4f} "
                    f"valid_acc={valid_acc:.4f} "
                    f"valid_f1={valid_f1:.4f}"
                )

                if valid_f1 > best_f1:
                    best_f1 = valid_f1

                    checkpoint = {
                        "arch": args.arch,
                        "model_state_dict": model.state_dict(),
                        "labels": labels,
                        "label_to_idx": label_to_idx,
                        "idx_to_label": idx_to_label,
                        "image_size": args.image_size,
                        "valid_f1": valid_f1,
                    }

                    ckpt_path = out_dir / f"best_{args.arch}.pt"
                    torch.save(checkpoint, ckpt_path)
                    print(f"Saved new best model to {ckpt_path} with valid_f1={valid_f1:.4f}")

            print(f"Best validation macro F1: {best_f1:.4f}")


        if __name__ == "__main__":
            main()

The script is driven entirely by command-line arguments, so you can point it at
your staged data and tune the run without editing code:

``--metadata_csv``
    The training metadata CSV from Part 1 — the one with the ``mel_path``,
    ``primary_label``, and ``split`` columns.
``--mel_root``
    Directory the ``mel_path`` entries are relative to: the spectrogram folder
    you staged onto Anvil.
``--arch``
    Any torchvision classification architecture (default ``efficientnet_b0``).
    Larger variants such as ``efficientnet_b5`` can score higher but cost more
    GPU memory and time.
``--out_dir``
    Where checkpoints are written (created if it doesn't exist).
``--epochs``
    Number of passes over the training set (default ``5``).
``--batch_size``
    Spectrograms per step (default ``32``). Lower it if you hit GPU
    out-of-memory errors; raise it if the GPU is underused.
``--lr``
    AdamW learning rate (default ``1e-4``).
``--num_workers``
    DataLoader processes that load and augment images in the background. Match
    this to the CPU cores you request in the Slurm script.
``--image_size``
    Side length the spectrograms are resized to before going into the model
    (default ``224``).

The Slurm Batch Script
----------------------

The batch script is an ordinary shell script with a block of ``#SBATCH``
directives at the top. Those lines aren't run by bash — Slurm reads them to
decide what resources to reserve and how to schedule the job. Everything after
them runs on the allocated compute node, which is where the ``python3`` call
actually trains the model on the GPU.

.. code-block:: shell

    #!/bin/bash
    #SBATCH --account cis260991
    #SBATCH --partition=gpu
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --gpus-per-node=1
    #SBATCH --mem=20G
    #SBATCH --job-name birdclef-train
    #SBATCH -t 01:30:00  # 1 hour 30 minutes

    python3 train.py --metadata_csv metadata_min5_recordings.csv --mel_root /anvil/projects/x-cis260991/melspecs/ --out_dir outputs/ --epochs 5 --batch_size 32 --lr 1e-4 --num_workers 8 --arch efficientnet_b5

The directives map directly onto what the scheduler reserves:

``--account``
    The allocation charged for the job — your ACCESS/Anvil project.
``--partition=gpu``
    Anvil's GPU partition. A CPU-only partition won't have a GPU to give you,
    so this is the one directive you can't omit for training.
``--nodes=1`` / ``--ntasks=1``
    A single process on a single node — all this script needs.
``--cpus-per-task=8``
    CPU cores for that process; they feed the DataLoader workers, so keep this
    in step with ``--num_workers``.
``--gpus-per-node=1``
    One GPU, which is all the single-GPU training loop uses.
``--mem=20G``
    Host (CPU) memory for the job.
``--job-name``
    A label that shows up in the queue, making your job easy to spot.
``-t 01:30:00``
    Walltime limit. Slurm kills the job if it runs past this, so pad it a
    little — but don't over-request, since a shorter request usually starts
    sooner.

Make sure the paths and options in the ``python3`` line match your setup: the
``--metadata_csv`` and ``--mel_root`` should point at the data you staged, and
``--num_workers`` should not exceed ``--cpus-per-task``.

Submitting and Tracking the Job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the script (e.g. ``train_job.sh``) and hand it to the scheduler:

.. code-block:: shell

    sbatch train_job.sh

Slurm prints a job ID and returns immediately — the job runs whenever the GPU
resources free up, whether or not you stay logged in. Two commands cover the
basics:

.. code-block:: shell

    squeue -u $USER        # your jobs waiting or running right now
    sacct -j <job_id>      # status and resource usage, including finished jobs

Anything the script prints — the per-epoch metrics and the "saved new best
model" lines — is captured to ``slurm-<job_id>.out`` in the directory you
submitted from. By default Slurm folds standard error into that same file; add
a ``#SBATCH --error=slurm-%j.err`` directive if you'd rather keep errors in a
separate ``slurm-<job_id>.err``. We pick the logs and checkpoints back up in
:doc:`monitoring`.