import numpy as np
import torch as th
import time
import glob
import matplotlib.pyplot as plt
import kernel_architecture.kernel_variables as kernel_variables
import kernel_architecture.kernel_net as kernel_net
import helper_functions as helpers


def run_testing(cfg):

    # Specify the paths for this script
    data_src_path = cfg["source_path"] + "/data/"
    model_src_path = cfg["source_path"] + "/distana_arch/saved_models/"

    # setting device on GPU if available, else CPU
    device = helpers.determine_device()

    # Set up the parameter and tensor classes
    params = kernel_variables.KernelParameters(
        cfg=cfg,
        device=device
    )
    # tensors1 = kernel_variables.KernelTensors(_params=params)
    tensors = kernel_variables.KernelTensors(params=params)

    # Initialize and set up the kernel network
    net = kernel_net.KernelNetwork(
        params=params,
        tensors=tensors
    )

    # Restore the network by loading the weights saved in the .pt file
    print('Restoring model (that is the network\'s weights) from file...')
    net.load_state_dict(th.load(model_src_path + "/" + cfg["model_name"] + "/" +
                                cfg["model_name"] + ".pt",
                                map_location=params.device))
    net.eval()

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad
    )
    print("Trainable model parameters:", pytorch_total_params)

    """
    TESTING
    """

    plt.style.use('dark_background')

    #
    # Set up the feed dictionary for the test iteration
    test_data_filenames = glob.glob(data_src_path + 'test/*')

    x = u'1'
    curr_idx = 0
    while x == u'1':

        error = 0.0
        std = 0.0
        for idx in range(10):

            time_start = time.time()

            # Evaluate the network for the given test data
            _, net_input, net_label, net_outputs = helpers.evaluate(
                net=net,
                data_filenames=test_data_filenames,
                params=params,
                tensors=tensors,
                _iter=curr_idx,
                testing=True
            )

            forward_pass_duration = time.time() - time_start
            print("\tForward pass took:", forward_pass_duration, "seconds.")

            net_outputs = net_outputs.detach().numpy()

            # Plot the wave activity
            fig, axes = plt.subplots(2, 2, figsize=[10, 8], sharex="all")
            for i in range(2):
                for j in range(2):
                    make_legend = True if (i == 0 and j == 0) else False
                    helpers.plot_kernel_activity(
                        ax=axes[i, j],
                        label=net_label,
                        net_out=net_outputs,
                        params=params,
                        make_legend=make_legend
                    )
            fig.suptitle('Model ' + cfg["model_name"], fontsize=12)
            print("i:%s" % (idx))
            if idx == 9:
                plt.show()

            # Visualize and animate the propagation of the 2d wave
            anim = helpers.animate_2d_wave(net_label, net_outputs, params)
            if idx == 9:
                plt.show()

            # Compute the error for only the closed loop steps
            diff = np.square(net_outputs[15:] - net_label[15:])
            mse = np.mean(diff)
            print("Error: %s" % (mse))
            error += mse
            std += np.std(diff)
            curr_idx += 1

        print("Average error over 10 sequences: %s" % (error / 10))
        print("Average Standard deviation over 10 sequences: %s" % (std / 10))
        # Retrieve user input to continue or quit the testing
        x = input("Press 1 to see another example, anything else to quit.")

    print('Done')
