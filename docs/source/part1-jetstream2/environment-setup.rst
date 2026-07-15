Setting Up the Environment
==========================

Jetstream2 instances come with minimal packages pre-installed on purpose as it allows the user to customize their environment. To explore our birdsong dataset, we will need to use Python. Let's first see if it is installed: 

```
$ which python3
/usr/bin/python3
```
Great! Python3 comes with the Ubuntu operating system, so it is pre-installed. 

However, other software for your own future analyses may not be preinstalled. Fortunately, unlike nearly all HPC and HTC systems, on the cloud we are able to directly install software including tools that require administrative privileges (e.g., `sudo`). 

To install the Python packages we need, let's first create a virtual python environment: 




   Create a reproducible Python environment on the VM (conda/mamba or venv +
   uv), install the ML/data libraries, and launch Jupyter for interactive work.

Installing Dependencies
-----------------------

.. todo:: Package manager choice and the dependency list.


Setting Up SSH Keys
-----------------------

.. todo:: Setup SSH keys for secure access to the VM and for transferring files between Jetstream2 and other resources.

Launching Jupyter
-----------------

.. todo:: Start Jupyter and connect to it (port forwarding / web desktop).
