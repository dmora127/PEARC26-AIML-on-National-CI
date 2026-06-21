Staging Data for Anvil
======================

.. todo::

   Get the processed dataset off Jetstream2 and ready for training on Anvil:
   choose a transfer method and verify integrity on arrival.

Transfer Options
----------------

.. todo:: scp/rsync, Globus, or object storage — recommend one and show it.

.. code-block::

    # Example scp command:
    rsync -avP /media/volume/birdclef-working-dir/mel-spectrograms/ x-user@anvil.rcac.purdue.edu:/anvil/projects/x-cis250844/pearc26-aiml/

This command will recursively copy the contents of the local directory containing the mel-spectrograms to the specified directory on Anvil, preserving file permissions and showing progress. Adjust the paths and username as needed for your specific setup. It is recommended to use rsync for large datasets as it can resume interrupted transfers and only copies changed files, making it more efficient than scp for subsequent transfers.

This transfer step will likely take some time, especially due to the large dataset, so be patient. It will be transferring ~500k files totalling ~100GB. Once the transfer is complete, you can verify that the files have arrived correctly on Anvil before proceeding to the next steps of training your model.

.. tip::
    If you are following along with the tutorial as part of the in-person workshop, we have already staged the data for you on Anvil. You can skip this section and move on to the next one, where we will start training our model.



Verifying the Transfer
----------------------

.. todo:: Checksums / record counts to confirm the data arrived intact.

Shelving your Jetstream2 Instance
---------------------------------

It is important to properly shut down your Jetstream2 instance to avoid unnecessary costs. Once you have verified that your data has been successfully transferred to Anvil, you can safely terminate your Jetstream2 instance. Make sure to double-check that all your important files and results have been saved and transferred before doing so, as terminating the instance will result in the loss of any data stored on it.

To shelve your Jetstream2 instance, follow these steps:

1. Log in to your Jetstream2 dashboard and locate the "CIS260991" Allocation associated with your instance..

.. image:: ../assets/jetstream2/jetstream2-exosphere-dashboard.png
    :alt: Jetstream2 Exosphere Dashboard
    :align: center
    :width: 80%

2. Locate your active "pearc26-aiml-on-national-ci" instance in the list of instances.

.. image:: ../assets/jetstream2/allocation-management.png
    :alt: Jetstream2 Exosphere Dashboard
    :align: center
    :width: 80%

3. Click on the instance to view its details.

4. Click the "Shelve" button to stop the instance and save its state. This will prevent any further charges while keeping your data intact.

.. image:: ../assets/jetstream2/instance-management.png
    :alt: Jetstream2 Exosphere Dashboard
    :align: center
    :width: 80%

5. If you need to access the instance again in the future, you can simply "Unshelve" it from the dashboard, and it will resume from where you left off.

.. warning::
    Make sure to shelve your instance if you are not actively using it, as leaving it running can incur unnecessary costs. Always double-check that your data has been safely transferred and backed up before shelving or terminating your instance.