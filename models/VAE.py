import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, z_dim):
        """Constructor

        Args:
            z_dim (int): Dimensions of the latent variable.

        Returns:
            None.

        Note:
            eps (float): Small amounts to prevent overflow and underflow.
        """
        super(VAE, self).__init__()
        self.eps = np.spacing(1)
        self.x_dim = 28 * 28 # The image in MNIST is 28×28
        self.z_dim = z_dim
        self.enc_fc1 = nn.Linear(self.x_dim, 400)
        self.enc_fc2 = nn.Linear(400, 200)
        self.enc_fc3_mean = nn.Linear(200, z_dim)
        self.enc_fc3_logvar = nn.Linear(200, z_dim)
        self.dec_fc1 = nn.Linear(z_dim, 200)
        self.dec_fc2 = nn.Linear(200, 400)
        self.dec_drop = nn.Dropout(0.2)
        self.dec_fc3 = nn.Linear(400, self.x_dim)

    def encoder(self, x):
        """Encoder

        Args:
            x (torch.tensor): Input data whose size is (Batch size, x_dim).

        Returns:
            mean (torch.tensor): Mean value of approximated posterior distribution whose size is (Batch size, z_dim).
            logvar (torch.tensor): Log-variance of approximated posterior distribution (Batch size, z_dim).
        """
        x = x.view(-1, self.x_dim)
        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        return self.enc_fc3_mean(x), self.enc_fc3_logvar(x)

    def sample_z(self, mean, log_var, device):
        """Sampling latent variables using reparametrization trick

        Args:
            mean (torch.tensor): Mean value of approximated posterior distribution whose size is (Batch size, z_dim).
            logvar (torch.tensor): Log-variance of approximated posterior distribution (Batch size, z_dim).
            device (String): "cuda" if GPU is available, or "cpu" otherwise.

        Returns:
            z (torch.tensor): Latent variable whose size is (Batch size, z_dim).
        """
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon * torch.exp(0.5 * log_var)

    def decoder(self, z):
        """Decoder

        Args:
            z (torch.tensor): Latent variable whose size is (Batch size, z_dim).

        Returns:
            (torch.tensor): Reconstruction data whose size is (batch size, x_dim).
        """
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = self.dec_drop(z)
        return torch.sigmoid(self.dec_fc3(z))

    def forward(self, x, device):
        """Forward propagation

        Args:
            x (torch.tensor): Input data whose size is (batch size, x_dim).
            device (String): "cuda" if GPU is available, or "cpu" otherwise.

        Returns:
            KL (torch.float): KL divergence
            reconstruction (torch.float): Reconstruction error
            z (torch.tensor): Latent variable whose size is (Batch size, z_dim).
            y (torch.tensor): Reconstruction data whose size is (batch size, x_dim).        
        """
        mean, log_var = self.encoder(x.to(device))
        z = self.sample_z(mean, log_var, device)
        y = self.decoder(z)
        KL = 0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))
        reconstruction = torch.sum(x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps))
        return [KL, reconstruction], z, y
