import torch
import torch.nn as nn

class LyapunovStabilizer(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.P = nn.Parameter(torch.eye(latent_dim), requires_grad=False)

    def get_energy(self, z):
        return torch.mm(torch.mm(z, self.P), z.t())
    
    def is_stable(self, z_current, z_next_pred_batch):
        v_curr = self.get_energy(z_current)

        # Efficient batched energy calculation
        P_z_next = torch.mm(z_next_pred_batch, self.P)
        v_next_batch = torch.sum(P_z_next * z_next_pred_batch, dim=1).view(-1, 1)

        stable_mask = (v_next_batch < v_curr).squeeze(1)

        return stable_mask
    
    def is_stable_batch(self, z_current, z_next_pred_batch):
        return self.is_stable(z_current, z_next_pred_batch)