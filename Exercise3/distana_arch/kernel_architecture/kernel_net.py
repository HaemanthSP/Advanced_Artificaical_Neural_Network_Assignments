import numpy as np
import torch as th
import torch.nn as nn
import kernel_architecture.prediction_kernel as prediction_kernel


class KernelNetwork(nn.Module):
    """
    This class contains the kernelized network topology for the spatio-temporal
    propagation of information
    """

    def __init__(self, params, tensors):

        super(KernelNetwork, self).__init__()

        self.params = params
        self.tensors = tensors

        #
        # Prediction Kernels

        # Initialize the shared Prediction Kernel (PK) network that will do the
        # PK calculations
        self.pk_net = prediction_kernel.PredictionKernelNet(params=params)

        # Initialize an adjacency matrix for the PK-TK connections
        self.pk_adj_mat = th.zeros(size=(2,
                                         params.pk_rows * params.pk_cols,
                                         params.pk_rows * params.pk_rows),
                                   device=params.device)

        # Define a dictionary that maps directions to numbers
        direction_dict = {"top": 1, "left top": 2, "left": 3, "left bottom": 4,
                          "bottom": 5, "right bottom": 6, "right": 7,
                          "right top": 8}

        # Running index to pass a distinct id to each PK
        pk_id_running = 0

        # Iterate over all PK rows and columns to create PK "instances"
        for pk_row in range(params.pk_rows):
            for pk_col in range(params.pk_cols):

                # Find the neighboring PKs to which this PK is connected
                neighbors = {"top": [pk_row - 1, pk_col],
                             "left top": [pk_row - 1, pk_col - 1],
                             "left": [pk_row, pk_col - 1],
                             "left bottom": [pk_row + 1, pk_col - 1],
                             "bottom": [pk_row + 1, pk_col],
                             "right bottom": [pk_row + 1, pk_col + 1],
                             "right": [pk_row, pk_col + 1],
                             "right top": [pk_row - 1, pk_col + 1]}

                # Set the values of the PK adjacency matrix on true that
                # represent a connection between the connected PKs
                for neighbor_direction in neighbors:

                    # Get the row and column index of the current neighbor
                    neighbor_row, neighbor_col = neighbors[neighbor_direction]

                    # If the neighbor lies within the defined field, define
                    # it as neighbor in the adjacency matrix
                    if (0 <= neighbor_row < params.pk_rows) and \
                       (0 <= neighbor_col < params.pk_cols):

                        # Determine the index of the neighbor
                        neighbor_idx = neighbor_row * params.pk_cols + \
                                       neighbor_col

                        # Set the corresponding entry in the adjacency matrix to
                        # one
                        self.pk_adj_mat[0, pk_id_running, neighbor_idx] = 1
                        self.pk_adj_mat[1, pk_id_running, neighbor_idx] = \
                            direction_dict[neighbor_direction]

                pk_id_running += 1

        #
        # Set up vectors that describe which lateral output goes to which
        # lateral input
        a = np.where(self.pk_adj_mat[0].cpu().detach().numpy() == 1)

        # PKs that are to be considered in the lateral update
        self.pos0 = th.from_numpy(a[0]).to(dtype=th.long)
        # PK lateral outputs that will be sent to the lateral inputs
        self.pos1 = th.from_numpy(a[1]).to(dtype=th.long)
        # PK lateral input neurons that will get inputs from the previous time
        # step's lateral output
        self.pos2 = (self.pk_adj_mat[1][a] - 1).to(dtype=th.long)

    def forward(self, dyn_in):
        """
        Runs the forward pass of all PKs in parallel for a given input
        :param dyn_in: The dynamic input for the PKs
        """

        # Write the dynamic PK input to the corresponding tensor
        if isinstance(dyn_in, th.Tensor):
            self.tensors.pk_dyn_in = dyn_in
        else:
            self.tensors.pk_dyn_in = th.from_numpy(
                dyn_in
            ).to(device=self.params.device)

        # Multiply the adjacency matrix with the lateral outputs to determine
        # which lateral outputs go to which lateral input
        mult = self.pk_adj_mat[0] * self.tensors.pk_lat_out.t()

        # Subselect only the relevant entries from the just created matrix to
        # omit zeros
        relevant_inputs = mult[self.pos0, self.pos1]

        # Set the appropriate lateral inputs to the lateral outputs from the
        # previous time step
        self.tensors.pk_lat_in[self.pos0, self.pos2] = relevant_inputs

        # Forward the PK inputs through the pk_net to get the outputs of these
        # PKs
        pk_dyn_out, pk_lat_out = self.pk_net.forward(
            dyn_in=self.tensors.pk_dyn_in,
            lat_in=self.tensors.pk_lat_in
        )

        # Update the output tensors of the PKs
        self.tensors.pk_dyn_out = pk_dyn_out
        self.tensors.pk_lat_out = pk_lat_out

    def reset(self, pk_num):
        self.tensors.reset(pk_num=pk_num)
