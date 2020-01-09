import torch as th


class NetworkParameters:
    """
    This class holds the parameters of the Network.
    """

    def __init__(self, cfg, device):

        #
        # System parameters
        self.device = device
        self.source_path = cfg["source_path"]

        #
        # Configuration parameters
        self.architecture_name = cfg["architecture_name"]
        self.model_name = cfg["model_name"]
        self.mode = cfg["mode"]
        self.save_model = cfg["save_model"]

        #
        # Training parameters
        self.epochs = cfg["epochs"]
        self.seq_len = cfg["seq_len"]

        #
        # Testing parameters
        self.teacher_forcing_steps = cfg["teacher_forcing_steps"]
        self.closed_loop_steps = cfg["closed_loop_steps"]

        #
        # Network parameters
        self.seq_len = cfg["seq_len"]

        self.input_size_x = 16
        self.input_size_y = 16

        self.tconv_num_channels = [1, 8, 1]
        self.tconv_kernel_size = 3


class NetworkTensors:
    """
    This class holds the tensors of the Network.
    """

    def __init__(self, params):
        self.params = params

        # Create the tensors by calling the reset method
        self.reset()

    def reset(self):

        #
        # Network tensors

        # Inputs
        self.inputs = th.zeros(size=(1,
                                     self.params.input_size_x,
                                     self.params.input_size_y),
                               device=self.params.device)

        # Outputs
        self.outputs = th.zeros(size=(1,
                                      self.params.input_size_x,
                                      self.params.input_size_y),
                                device=self.params.device)
