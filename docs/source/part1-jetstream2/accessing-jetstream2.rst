Accessing Jetstream2
====================

Logging In
----------

1. Navigate to the Jetstream2 Exosphere user portal: https://jetstream2.exosphere.app/exosphere/

2. Click the "Add Allocation" box.

3. On the new page, click the red "Add ACCESS Account" button and authenticate
   with your ACCESS credentials. This will redirect you to log in via the
   institutional account you created your ACCESS ID with.

4. After authenticating, you will see the option to choose an allocation and
   should see the following listed: "CIS260991 — Training: AI/ML Workflows
   Across National Cyberinfrastructure." Select it and click "Choose".

5. From the list of available regions, select any of the available
   institutions and click "Choose".

Launching an Instance
---------------------

Once you have logged in and successfully accessed the tutorial's resource
allocation, you can launch an instance.

1. Select the pane showing the course allocation.

   At this point, it is possible to view the resources available to our ACCESS
   allocation on Jetstream2. Should you choose to request your own ACCESS
   allocation, you will have access to this same resource utilization view. It
   is helpful for understanding how efficiently you are using your resources
   and when you may need to request an additional allocation to avoid an
   interruption to your work.

2. In the upper right corner, click "Create" and select "Instance" from the
   dropdown menu.

   In the Jetstream2 cloud, you have the option of customizing your own
   instance — this means selecting your own operating system (e.g., Ubuntu,
   Red Hat), operating system version, and more. This ability to customize
   aspects of the virtual machine your computations will run on is what makes
   a cloud environment ideal for testing and exploring new software and
   datasets. We will explore more of this customization in the next step.

3. Select the "By Image" tab and in the search bar, type "PEARC26-AIML-Part1".
   You should see the image appear in the list below. Click on "Create Instance"
   to start a new instance using this image. This image has been pre-configured with the
   software and dependencies needed for this workshop.

4. In the "Create Instance" page, complete the setup as follows:

    - Name your instance ``<firstname-lastname>-birdclef-2026``.
      (For example, if your name is John Doe, you would name your instance ``john-doe-birdclef-2026``.)
    - Flavor select ``g3.large``.
    - Select ``Yes`` for ``Enable web desktop``.

   These options include different GPU architectures, CPU and RAM allocations,
   disk space, and more. You can explore the Advanced Options, but for now, do
   not change these settings.

4. After you have named your instance, leave the remaining settings at their
   default values and click "Create" to launch your instance. This will take a few minutes to complete. You will be redirected back to the allocation page where you can monitor the status of your instance. Once it is ready, you will be able to connect to it and begin working.

Creating and Attaching a Storage Volume
---------------------------------------

We also need to create a storage volume to hold our dataset and any other files we may want to save. This storage volume will be attached to our instance and will persist even after the instance is terminated. This means that we can stop and start our instance without losing any of our data. We will use this storage volume as a scratch space for our work during the workshop.

To create a storage volume, follow these steps:

1. Go to the ``CIS260991`` allocation page and click on the "Volumes" tab.

2. Click on the "Create Volume" button.

3. In the "Create Volume" page, give your volume a name and select the size of the volume. For this workshop, we will use the default size of 200 GB. Name you volume ``birdclef-2026`` and click "Create".

4. Wait for the volume to be created and become ``available``. This may take a few minutes.

5. Once the volume is available, click on the "Attach" button and select **your** instance from the dropdown menu. Click "Attach" to attach the volume to your instance.


Connecting to Your VM
---------------------

After creating your instance, you will be redirected back to the resource
utilization overview page. In the "Instances" pane, you will notice your
instance is in the "Building" state — this means your virtual machine is being
provisioned. Your instance will then show "Running Setup", and finally
"Ready". This process may take several minutes or more depending on the
instance size and current demand on the system.

1. Once your instance is "Ready", click on its name.

2. You will be redirected to a new page. In the "Connect to" dropdown box,
   select "Console".

Once you see a terminal window, you will have successfully launched your first
Jetstream2 instance!
