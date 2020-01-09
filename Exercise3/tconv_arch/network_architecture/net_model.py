import torch as th
import torch.nn as nn
import network_architecture.tconv as tconv


class Model(nn.Module):
    """
    This class contains the CNN network topology for the spatio-temporal
    propagation of information
    """

    def __init__(self, params, tensors):

        super(Model, self).__init__()

        self.params = params
        self.tensors = tensors

        #
        # Set up the model

        # Temporal convolution network
        self.tconv = tconv.TemporalConvNet(
            num_inputs=1,
            num_channels=self.params.tconv_num_channels,
            kernel_size=self.params.tconv_kernel_size,
            dropout=0.0
        ).to(device=self.params.device)

    def forward(self, net_in):
        """
        Runs the forward pass of the CNN network for a given input
        :param net_in: The input for the network
        """

        # Convert the net_in numpy array to a tensor to feed it to the network
        net_in_tensor = th.from_numpy(net_in).to(device=self.params.device)

        # Forward the input through the conv1 layer
        tconv_out = self.tconv(net_in_tensor)

        # Update the output and hidden state tensors of the network
        self.tensors.output = tconv_out

    def reset(self):
        self.tensors.reset()
