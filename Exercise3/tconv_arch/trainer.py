"""
In this script, a neural network model is trained to simulate the propagation
of a wave by using the PyTorch library.
"""

import numpy as np
import torch as th
import torch.nn as nn
import glob
import time
import matplotlib.pyplot as plt
from threading import Thread
import network_architecture.net_variables as net_variables
import network_architecture.net_model as net_model
import helper_functions as helpers


def run_training(cfg):

    # Read the configuration file to be able to save it later if desired
    if cfg["save_model"]:
        with open(cfg["source_path"] + "/tconv_arch/main.py", "r") as f:
            cfg_file = f.read()

    # Print some information to console
    print("Architecture name:", cfg["architecture_name"])
    print("Model name:", cfg["model_name"])

    # Specify the paths for this script
    data_src_path = cfg["source_path"] + "/data/"
    model_src_path = cfg["source_path"] + "/tconv_arch/saved_models/"

    # Set device on GPU if specified so in the main file, else CPU
    device = helpers.determine_device()

    # Set up the parameter and tensor classes
    params = net_variables.NetworkParameters(
        cfg=cfg,
        device=device
    )
    tensors = net_variables.NetworkTensors(params=params)

    # Initialize and set up the network model
    net = net_model.Model(
        params=params,
        tensors=tensors
    )

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    #
    # Set up the optimizer and the criterion (loss)
    optimizer = th.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    criterion = nn.MSELoss()

    #
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    epoch_errors_val = []

    best_train = np.infty
    best_val = np.infty

    #
    # Get the training and validation file names
    train_data_filenames = np.sort(glob.glob(data_src_path + 'train/*'))
    val_data_filenames = np.sort(glob.glob(data_src_path + 'val/*'))

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if cfg["continue_training"]:
        print('Restoring model (that is the network\'s weights) from file...')
        net.load_state_dict(th.load(model_src_path + "/" + cfg["model_name"]
                                    + "/" + cfg["model_name"] + ".pt"))
        net.eval()

    """
    TRAINING
    """

    a = time.time()

    #
    # Start the training and iterate over all epochs
    for epoch in range(cfg["epochs"]):

        epoch_start_time = time.time()

        # Shuffle the training_data_filenames at the beginning of each epoch to
        # have variable (stochastic) training data orders
        np.random.shuffle(train_data_filenames)

        sequence_errors = []

        # Iterate over all training iterations and evaluate the network
        for train_iter in range(len(train_data_filenames)):

            # Evaluate and train the network for the given training data
            mse, _, _, _ = helpers.evaluate(
                net=net,
                data_filenames=train_data_filenames,
                params=params,
                tensors=tensors,
                criterion=criterion,
                optimizer=optimizer,
                _iter=train_iter
            )

            sequence_errors.append(mse.item())

        epoch_errors_train.append(np.mean(sequence_errors))

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors_train[-1]

        #
        # Compute validation error

        # Evaluate and validate the network for the given validation data
        mse, _, _, _ = helpers.evaluate(
            net=net,
            data_filenames=val_data_filenames,
            params=params,
            tensors=tensors,
            criterion=criterion,
            _iter=0
        )

        epoch_errors_val.append(mse.item())

        # Save the model to file (if desired)
        if cfg["save_model"] and mse.item() < best_val:
            # Start a separate thread to save the model
            thread = Thread(target=helpers.save_model_to_file(
                model_src_path=model_src_path,
                cfg_file=cfg_file,
                epoch=epoch,
                epoch_errors_train=epoch_errors_train,
                epoch_errors_val=epoch_errors_val,
                net=net,
                params=params))
            thread.start()

        # Create a plus or minus sign for the validation error
        val_sign = "(-)"
        if epoch_errors_val[-1] < best_val:
            best_val = epoch_errors_val[-1]
            val_sign = "(+)"

        #
        # Print progress to the console
        print('Epoch ' + str(epoch + 1).zfill(int(np.log10(cfg["epochs"])) + 1)
              + '/' + str(cfg["epochs"]) + ' took '
              + str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')
              + ' seconds.\t\tAverage epoch training error: ' + train_sign
              + str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')
              + '\t\tValidation error: ' + val_sign
              + str(np.round(epoch_errors_val[-1], 10)).ljust(12, ' '))

    b = time.time()
    print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')

    """
    BRIEF TESTING
    """

    plt.style.use('dark_background')

    #
    # Set up the feed dictionary for the test iteration
    test_data_filenames = glob.glob(data_src_path + 'test/*')

    x = u'1'
    curr_idx = 0
    while x == u'1':

        _, net_input, net_label, net_outputs = helpers.evaluate(
            net=net,
            data_filenames=test_data_filenames,
            params=params,
            tensors=tensors,
            criterion=criterion,
            _iter=curr_idx
        )

        net_outputs = net_outputs.cpu().detach().numpy()

        fig, axes = plt.subplots(2, 2, figsize=[10, 8])
        for i in range(2):
            for j in range(2):
                make_legend = True if (i == 0 and j == 0) else False
                helpers.plot_kernel_activity(
                    ax=axes[i, j],
                    label=net_label,
                    net_out=net_outputs,
                    net_in=net_input,
                    params=params,
                    make_legend=make_legend
                )

        fig.suptitle('Optimization', fontsize=12)

        plt.show()
        x = input("Press 1 to see another example, anything else to quit.")
        curr_idx += 1

    print('Done')
