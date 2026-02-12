import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, gamma=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def forward(self, z1, z2):

        # Invariance loss
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)

        var_loss = torch.mean(F.relu(self.gamma - std_z1)) + \
                   torch.mean(F.relu(self.gamma - std_z2))

        # Covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        cov_z1 = (z1.T @ z1) / (z1.size(0) - 1)
        cov_z2 = (z2.T @ z2) / (z2.size(0) - 1)

        cov_loss = off_diagonal(cov_z1).pow(2).sum() / z1.size(1) + \
                   off_diagonal(cov_z2).pow(2).sum() / z2.size(1)

        loss = (
            self.sim_coeff * sim_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        return loss
