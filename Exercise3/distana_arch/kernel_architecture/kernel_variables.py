import torch as th


class KernelParameters:
    """
    This class holds the parameters of the Kernel Network.
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
        # PK specific parameters
        self.pk_rows = 16
        self.pk_cols = 16
        self.pk_neighbors = 8  # Each PK has per definition eight neighbors
        self.pk_batches = self.pk_rows * self.pk_cols

        # Input sizes (neurons)
        self.pk_dyn_in_size = 1
        self.pk_lat_in_size = 1

        # Output sizes (neurons)
        self.pk_dyn_out_size = 1
        self.pk_lat_out_size = 1


class KernelTensors:
    """
    This class holds the tensors of the Kernel Network.
    """

    def __init__(self, params):
        self.params = params

        # Initialize the tensors by calling the reset method (this may not be
        # clean code style, yet it spares lots of lines :p)
        self.reset(self.params.pk_batches)

    def reset(self, pk_num):

        #
        # PK tensors

        # Inputs
        self.pk_dyn_in = th.zeros(size=(pk_num,
                                        self.params.pk_dyn_in_size),
                                  device=self.params.device)
        self.pk_lat_in = th.zeros(
            size=(pk_num,
                  self.params.pk_lat_in_size * self.params.pk_neighbors),
            device=self.params.device
        )

        # Outputs
        self.pk_dyn_out = th.zeros(size=(pk_num,
                                         self.params.pk_dyn_out_size),
                                   device=self.params.device)
        self.pk_lat_out = th.zeros(size=(pk_num,
                                         self.params.pk_lat_out_size),
                                   device=self.params.device)
