Accessing Jetstream2
====================

.. todo::

   Walk through logging into Exosphere/Horizon, selecting an allocation, and
   launching an instance (flavor/image choice, sizing for this workload).

Logging In
----------

(1) Navigate to the Jetstream2's Exosphere user portal here: https://jetstream2.exosphere.app/exosphere/

(2) Click the "Add Allocation" box. 

(3) On the new page, click the red "Add ACCESS account" button and authenticate with your ACCESS credentials. This will redirect you to login via the institution account you created your ACCESS ID with. 

(4) After authenticating, you will see the option to choose an allocation and should see the following listed: "CIS260991 — Training: AI/ML Workflows Across National Cyberinfrastructure." Select it, and click "Choose".

5) From the list of availabe regions, select any of the available institutions and click "Choose".  



Launching an Instance
---------------------
Once you have logged in and successfully accessed the tutorial's resource allocation, you can launch an instance. 

(1) Select the pane showing the course allocation. 

At this point, is it possible to view the resources available to our ACCESS allocation on Jestream2. Should you choose to request your own ACCESS allocation, you will have access to this same resource utalization view. It is helpful for understanding how efficently you are using your resources and when you may need to request an additional allocation to avoid an inturruption to your work. 

(2) In the upper right corner, click "Create" and select "Instance" from the dropdown menu. 

In the Jetstream2 cloud, you have the option of cusotmizing your own instance - this means selecting your own operating system (e.g., Ubuntu, Red Hat), operating system version, and more. This ability to customize aspects of the virtual machine you compute will run on is what makes a cloud environment ideal for testing and exploring new software and datsets. We will explore more of this customization in the next step. 

(3) From the Ubuntu pane, select 24.04 (latest) to launch your instance. On the new page, explore the options available to your session. For today's workshop, we will use the default options and do not need to change any settings; however, please name your instance.

These options include different GPU architectures, CPU and RAM allocations, disk space, and more. You can explore Advanced Options, but for now, do not change these settings.

(4) Leaving the default values and after you have named your instance, click "Create" to launch your instance. 

.. todo:: Choose image, flavor, and storage; create the instance.

Connecting to Your VM
---------------------
After creating your instance, you will be redirected back to the resource utalization overview page. In the "Volumes" pane, you will notice your instance is in the "Building" state - this means your virtual machine is being customized. Your instance will then show "Running Setup", and finally "Ready". This process may take several minutes or more depending on ______. 

(1) Once your instance is "Ready", click on its name. 
(2) You will be redirected to a new page. In the "Connect to" dropdown box, select "Console". 

Once you see a terminal window, you will have successfully launched your first Jetstream2 instance!


.. todo:: Web shell vs. SSH; how to connect and verify access.
