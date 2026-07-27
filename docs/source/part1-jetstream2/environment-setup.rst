Setting Up the Environment
==========================

You have an instance running and a volume of data attached from
:doc:`accessing-jetstream2`, but the VM itself is deliberately bare. This
section builds the working environment the rest of Part 1 depends on: a Python
virtual environment holding pinned versions of the audio, machine-learning, and
plotting libraries, plus a Jupyter server you can reach from your own browser
for the interactive work in :doc:`data-exploration`. Setting it up by hand is
part of the point — choosing your own software stack is exactly the freedom a
cloud VM gives you over the shared, module-managed environments you'll meet on
Anvil and the OSPool.

Installing Dependencies
-----------------------

Jetstream2 instances come with minimal packages pre-installed on purpose, as it
allows the user to customize their environment. To explore our birdsong dataset,
we will need to use Python. Let's first see if it is installed:

.. code-block:: console

    $ which python3
    /usr/bin/python3

Great! Python3 comes with the Ubuntu operating system, so it is pre-installed.

However, other software for your own future analyses may not be preinstalled.
Fortunately, unlike nearly all HPC and HTC systems, on the cloud we are able to
directly install software, including tools that require administrative
privileges (e.g., ``sudo``).

To install the Python packages we need, let's first create a virtual Python
environment:

.. code-block:: shell

    # Create a virtual environment for Python 3.12 (or 3.9 if you prefer)
    python3 -m venv birdclef-venv
    source birdclef-venv/bin/activate

Install the required packages for the preprocessing pipeline (everything except
inference) from the ``requirements-preprocessing.txt`` file. This will install
packages for audio I/O and analysis, voice activity detection, spectrogram image
generation, metadata handling, and progress bars:

.. code-block:: shell

    # Upgrade pip and install the required packages
    pip install --upgrade pip
    pip install -r requirements-preprocessing.txt

.. collapse:: Code: requirements-preprocessing.txt

    .. code-block:: text

        # Requirements for the preprocessing pipeline (everything except inference):
        #   speech_separator.py            (Silero VAD -> speech/non-speech regions)
        #   detect_calls.py                (energy-based call detection)
        #   generate_spectrograms*.py      (mel spectrogram PNG generation)
        #   make_effnet_training_csv*.py   (train/val split metadata for EfficientNet)
        #   merge_mel_with_train_metadata.py
        #   preprocessing.py
        #
        #   python3 -m pip install -r requirements-preprocessing.txt
        #
        # Pins target Python 3.12 (all ship prebuilt wheels -- nothing compiles from
        # source). The original anaconda env (Python 3.9.13, `module load anaconda`)
        # used older versions; those are noted in comments. If you install under
        # Python 3.9, use the 3.9 versions instead -- the 3.12 pins below will not all
        # resolve there.
        #
        # NOTE: torch/torchaudio are only used to run the Silero VAD model (CPU is
        # fine). On a machine without a matching CUDA runtime, the default PyPI wheels
        # can pull a CUDA torchaudio build that fails at import with
        # "libcudart.so.NN: cannot open shared object file". To avoid this, install
        # torch + torchaudio from the CPU wheel index BEFORE running this file:
        #
        #   pip install --index-url https://download.pytorch.org/whl/cpu \
        #       torch==2.8.0 torchaudio==2.8.0
        #   pip install -r requirements-preprocessing.txt
        #
        # For a specific CUDA build instead, install matching torch/torchaudio wheels
        # from https://pytorch.org first, then run this file.

        # --- audio I/O and analysis ---
        librosa==0.11.0
        soundfile==0.13.1
        numpy==1.26.4          # py3.9 env: 1.24.4
        scipy==1.13.1          # py3.9 env: 1.9.1

        # --- voice activity detection (speech_separator.py) ---
        silero-vad==6.2.1
        torch==2.8.0
        torchaudio==2.8.0      # required by silero-vad; must match torch version

        # --- spectrogram image generation ---
        matplotlib==3.9.2      # py3.9 env: 3.5.2
        pillow==10.4.0         # py3.9 env: 9.2.0

        # --- metadata / train-val splitting ---
        pandas==2.2.3          # py3.9 env: 1.4.4
        scikit-learn==1.5.2    # py3.9 env: 1.6.1

        # --- progress bars ---
        tqdm==4.66.5           # py3.9 env: 4.64.1

Launching Jupyter
-----------------

Prerequisites:

- Instance is running and reachable via SSH or the Exosphere web shell.
- Ability to copy URLs from the terminal into your local browser.
- Rocky9 only: ``sudo`` access to update the firewall the first time you open
  the port.

Use this when you:

- Did not enable (or do not want) the Web Desktop.
- Prefer to connect from your own local browser directly.
- Are working over a plain SSH session or the Exosphere web shell.

How it works:

We provide a helper script, ``jupyter-ip.sh``, that discovers the VM's public IP
and injects it into the Jupyter URL. A VM does not inherently know its public
IP, so this script saves you from manually editing the URL.

Steps:

#. Open an Exosphere web shell OR SSH into the instance from your local machine.
#. Load the module for your OS image.
#. ``cd`` into the ``/`` root directory to mount volumes correctly.
#. Run ``jupyter-ip.sh`` (it launches Jupyter and prints a URL with a security
   token).
#. Copy the final line containing the public IP (e.g.,
   ``http://149.165.xxx.xxx:8888/?token=...``) into your local machine's
   browser.
