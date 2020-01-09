import numpy as np
import torch as th
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def set_up_batch(_iter, data_filenames, params):
    """
    In this function, a batch is composed to be fed into the network.
    :param _iter: The iteration of the current epoch
    :param data_filenames: The paths to the data files
    :param params: The parameters of the model
    :return: Two lists: one with network inputs, and another one with
             corresponding labels
    """

    # Determine which inputs of the 16x16 default input field are the center for
    # this pk_rows and pk_cols set up
    data_xmin = int(8 - np.floor(params.pk_cols / 2.))
    data_xmax = int(8 + np.ceil(params.pk_cols / 2.))
    data_ymin = int(8 - np.floor(params.pk_rows / 2.))
    data_ymax = int(8 + np.ceil(params.pk_rows / 2.))

    # Get width and height of the visual field of the network
    width, height = data_xmax - data_xmin, data_ymax - data_ymin

    # Load data from file
    data = np.load(data_filenames[_iter])[:params.seq_len + 1]

    # Generate a random position for the visual field
    x = np.random.randint(0, max(1, 15 - width))
    y = np.random.randint(0, max(1, 15 - height))

    # Sub select only the data values that are of interest
    data = data[:, :, x:x + width, y:y + height]

    # Get first and second dimension of data
    dim0, dim1 = np.shape(data)[:2]

    # In case the desired field is larger than 16x16 pixels (which is the data
    # dimension), pad the data with zeros
    if params.pk_rows >= 16 or params.pk_cols >= 16:
        data = np.pad(
            array=data,
            pad_width=((0, 0),
                       (0, 0),
                       (0, params.pk_rows - 16),
                       (0, params.pk_cols - 16)),
            mode="constant",
            constant_values=0
        )

    # Reshape the data array to have the kernels on one dimension
    data = np.reshape(data, [dim0, dim1, params.pk_rows * params.pk_cols])

    # Swap the second and third dimension of the data
    data = np.swapaxes(data, axis1=1, axis2=2)

    # Split the data into inputs and labels
    _net_input = np.array(data[:-1], dtype=np.float32)
    _net_label = np.array(data[1:, :, 0:1], dtype=np.float32)

    return _net_input, _net_label


def evaluate(net, data_filenames, params, tensors, criterion=None,
             optimizer=None, _iter=0, testing=False):
    """
    This function evaluates the network for given data and optimizes the weights
    if an optimizer is provided.
    :param net: The network
    :param data_filenames: The filenames where the data to forward are lying
    :param params: The parameters of the network
    :param tensors: The tensors of the network
    :param criterion: The criterion to measure the error
    :param optimizer: The optimizer to optimize the weights
    :param _iter: The current iteration of e.g. the training
    :param testing: Bool that determines weather network is being tested
    :return: The error, net inputs, net labels and net outputs
    """

    seq_len = params.seq_len if not testing\
        else params.teacher_forcing_steps + params.closed_loop_steps

    # Generate the training data batch for this iteration
    net_input, net_label = set_up_batch(
        _iter=_iter,
        data_filenames=data_filenames,
        params=params
    )

    # Set up an array of zeros to store the network outputs
    net_outputs = th.zeros(size=(seq_len,
                                 params.pk_batches,
                                 params.pk_dyn_out_size))

    if optimizer:
        # Set the gradients back to zero
        optimizer.zero_grad()

    # Reset the network to clear the previous sequence
    net.reset(pk_num=params.pk_batches)

    # Iterate over the whole sequence of the training example and perform a
    # forward pass
    for t in range(seq_len):

        # Prepare the network input for this sequence step
        if testing and t > params.teacher_forcing_steps:
            # Closed loop - receiving the output of the last time step as input
            dyn_net_in_step = net_outputs[t-1].detach().numpy()
        else:
            dyn_net_in_step = net_input[t, :, :params.pk_dyn_out_size]

        # Forward the input through the network
        net.forward(dyn_in=dyn_net_in_step)

        # Store the output of the network for this sequence step
        net_outputs[t] = tensors.pk_dyn_out

    mse = None

    if criterion:
        # Get the mean squared error from the evaluation list
        mse = criterion(net_outputs, th.from_numpy(net_label))
        # Alternatively, the mse can be calculated 'manually'
        # mse = th.mean(th.pow(net_outputs - th.from_numpy(net_label), 2))

        if optimizer:
            mse.backward()
            optimizer.step()

    return mse, net_input, net_label, net_outputs


def determine_device():
    """
    This function evaluates whether a GPU is accessible at the system and
    returns it as device to calculate on, otherwise it returns the CPU.
    :return: The device where tensor calculations shall be made on
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(th.cuda.get_device_name(0))
        print("Memory Usage:")
        print("\tAllocated:",
              round(th.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("\tCached:   ", round(th.cuda.memory_cached(0) / 1024 ** 3, 1),
              "GB")
        print()
    return device


def save_model_to_file(model_src_path, cfg_file, epoch, epoch_errors_train,
                       epoch_errors_val, net, params):
    """
    This function writes the model weights along with the network configuration
    and current performance to file.
    :param model_src_path: The source path where the model will be saved to
    :param cfg_file: The configuration file
    :param epoch: The current epoch
    :param epoch_errors_train: The training epoch errors
    :param epoch_errors_val: The validation epoch errors
    :param net: The actual model
    :param params: The parameters of the model
    :return: Nothing
    """
    # print("\nSaving model (that is the network's weights) to file...")

    _model_save_path = model_src_path + "/" + params.model_name + "/"
    if not os.path.exists(_model_save_path):
        os.makedirs(_model_save_path)

    # Save model weights to file
    th.save(net.state_dict(), _model_save_path + params.model_name + ".pt")

    output_string = cfg_file + "\n#\n# Performance\n\n"

    output_string += "CURRENT_EPOCH = " + str(epoch) + "\n"
    output_string += "EPOCHS = " + str(params.epochs) + "\n"
    output_string += "CURRENT_TRAINING_ERROR = " + \
                     str(epoch_errors_train[-1]) + "\n"
    output_string += "LOWEST_TRAINING_ERROR = " + \
                     str(min(epoch_errors_train)) + "\n"
    output_string += "CURRENT_VALIDATION_ERROR = " + \
                     str(epoch_errors_val[-1]) + "\n"
    output_string += "LOWEST_VALIDATION_ERROR = " + \
                     str(min(epoch_errors_val))

    # Save the configuration and current performance to file
    with open(_model_save_path + 'cfg_and_performance.txt', 'w') as _text_file:
        _text_file.write(output_string)


def plot_kernel_activity(ax, label, net_out, params, net_in=None,
                         make_legend=False):
    """
    This function displays the wave activity of a single kernel.
    :param ax: The plot where the activity shall be displayed in
    :param label: The label for the wave (ground truth)
    :param net_out: The network output
    :param params: The parameters of the model
    :param net_in: The network input
    :param make_legend: Boolean that indicates weather a legend shall be created
    """

    central_kernel = (params.pk_rows * params.pk_cols) // 2

    if net_in is not None:
        ax.plot(range(len(net_in)), net_in[:, central_kernel, 0],
                label='Network input', color='green')
    ax.plot(range(len(label)), label[:, central_kernel, 0],
            label='Target', color='deepskyblue')
    ax.plot(range(len(net_out)), net_out[:, central_kernel, 0],
            label='Network output', color='red', linestyle='dashed')
    if net_in is None:
        yticks = ax.get_yticks()[1:-1]
        ax.plot(np.ones(len(yticks)) * params.teacher_forcing_steps, yticks,
                color='white', linestyle='dotted',
                label='End of teacher forcing')
    if make_legend:
        ax.legend()


def animate_2d_wave(net_label, net_outputs, params):
    """
    This function visualizes the spatio-temporally expanding wave
    :param net_label: The corresponding labels
    :param net_outputs: The network output
    :param params: The parameters of the model
    :return: The animated plot of the 2d wave
    """

    # Bring the data into a format that can be displayed as heatmap
    data = np.reshape(net_outputs, [len(net_outputs),
                                    params.pk_rows,
                                    params.pk_cols,
                                    len(net_outputs[0, 0])])
    net_label = np.reshape(net_label, [len(net_label),
                                       params.pk_rows,
                                       params.pk_cols,
                                       len(net_label[0, 0])])

    # Define a grid size that shall be visualized
    gs1 = 0
    gs2 = params.pk_rows

    # First set up the figure, the axis, and the plot element we want to
    # animate
    fig, axes = plt.subplots(1, 2, figsize=[12, 6], dpi=100)
    im1 = axes[0].imshow(net_label[20, gs1:gs2, gs1:gs2, 0], cmap='Blues')

    # Visualize the obstacle if there is one
    txt1 = axes[0].text(0, axes[0].get_yticks()[0], 't = 0', fontsize=20,
                        color='white')
    axes[0].set_title("Network Output")

    # In the subfigure on the right hand side, visualize the true data
    im2 = axes[1].imshow(net_label[20, gs1:gs2, gs1:gs2, 0], cmap='Blues')
    axes[1].set_title("Ground Truth")

    anim = animation.FuncAnimation(fig, animate, frames=len(data),
                                   fargs=(params.teacher_forcing_steps, data,
                                          im1, im2, txt1, gs1, gs2, net_label),
                                   interval=1)

    return anim


def animate(_i, _teacher_forcing_steps, _data, _im1, _im2, _txt1, _gs1, _gs2,
            _net_label):

    # Pause the simulation briefly when switching from teacher forcing to
    # closed loop prediction
    if _i == _teacher_forcing_steps:
        time.sleep(1.0)
    elif _i < 150:
        time.sleep(0.05)

    # Set the pixel values of the image to the data of timestep _i
    _im1.set_array(_data[_i, _gs1:_gs2, _gs1:_gs2, 0])
    if _i < len(_net_label) - 1:
        _im2.set_array(_net_label[_i, _gs1:_gs2, _gs1:_gs2, 0])

    # Display the current timestep in text form in the plot
    if _i < _teacher_forcing_steps:
        _txt1.set_text('Teacher forcing, t = ' + str(_i))
    else:
        _txt1.set_text('Closed loop prediction, t = ' + str(_i))

    return _im1, _im2
