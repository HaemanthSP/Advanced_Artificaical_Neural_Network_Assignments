"""
This file contains the code of a Beta Variational Autoencoder. It can be used
for loading a pre-trained model (that is the network weights) and decoding any
desired 32-dimensional latent tensor. Thus, by feeding a tensor of size (1, 32)
through the decoder, a 64x64 image will be reconstructed from that latent
tensor.
"""

import torch as th
import torch.nn as nn
from torch.autograd import Variable


def reparameterize(mu, logvar):
    """
    This function transforms a mu and logvar (variance) tensor into one tensor,
    based on mu and logvar
    """
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    """
    This class can be used for reshaping a tensor in a forward pass in place. As
    such, it can be used conveniently in PyTorch's "Sequential" class.
    """
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE(nn.Module):
    """
    This class contains the Beta Variational Autoencoder layers, along with
    appropriate calls for the encoder and decoder part.
    """

    def __init__(self, filename):
        super(BetaVAE, self).__init__()

        self.z_dim = 32  # Size of the hidden latent space
        self.nc = 3  # Number of channels (RGB image -> 3 channels)

        # Define the encoder network, using convolutions and ReLu functions
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            View((-1, 256*1*1)),
            nn.Linear(256, self.z_dim*2),
        )

        # Define the decoder network, using transposed convolutions and ReLus
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            View((-1, 256, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.nc, 4, 2, 1),
        )

        # Load the pre-trained weights from file and apply them to the model
        weights = th.load(filename)
        self.load_state_dict(weights)

    def forward(self, x):
        """
        Forward an image x through the encoder to compress it and subsequently
        feed the latent compression through the encoder to obtain an image
        again. Image dimensions must be [batch_size, 3, 64, 64].
        """
        distributions = self.encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparameterize(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, mu, logvar

    def encode(self, x):
        """
        Calls the encoder for a given input x. Dimensionality of x - which is an
        image - is supposed to be [batch_size, 3, 64, 64].
        Returns a compressed latent code z of the input image x.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Calls the decoder for a given latent code z. Dimensionality of z - which
        is the latent code - is supposed to be [batch_size, 32].
        Returns a reconstructed image from the compressed latent code z.
        """
        return self.decoder(z)


##########
# SCRIPT #
##########

# Load the model
model = BetaVAE(filename="weights.dat")

# TODO: (a) Create a latent state tensor and forward it through the decoder
#		to reconstruct an image from that latent code

# TODO: (b) Identify at least four neurons in the latent tensor z that have a
#		distinct effect on the generated images
