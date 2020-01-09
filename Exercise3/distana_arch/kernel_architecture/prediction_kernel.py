import torch as th
import torch.nn as nn


class PredictionKernelNet(nn.Module):
    """
    This class represents the shared-weights-network for all Prediction Kernels
    """

    def __init__(self, params):

        super(PredictionKernelNet, self).__init__()

        self.params = params

        # TODO: Define reasonable layers (weights) here

        # Initialize very simple dummy weights
        self.dummy_weights = nn.Linear(
            in_features=params.pk_dyn_in_size +
                        (params.pk_lat_in_size * params.pk_neighbors),
            out_features=params.pk_dyn_out_size + params.pk_lat_out_size,
            bias=True
        ).to(device=self.params.device)

    def forward(self, dyn_in, lat_in):

        # TODO: Add the forward pass of the defined layers here

        # Concatenate the dynamic and lateral input into one tensor
        combined_input = th.cat(tensors=(dyn_in, lat_in), dim=1)

        # Forward the combined input tensor through the dummy layer and apply a
        # tanh function
        post_act = th.tanh(self.dummy_weights(combined_input))

        # Dynamic output
        dyn_out = post_act[:, :self.params.pk_dyn_out_size]

        # Lateral output
        lat_out = post_act[:, self.params.pk_dyn_out_size:]

        return dyn_out, lat_out
