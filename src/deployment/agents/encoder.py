import torch.nn as nn
import torch

class EventEncoder(nn.Module):
    def __init__(self, input_dim=13, latent_dim=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, event_list):
        # Check if we are processing a Batch (3D) or Single Item (2D)
        is_batched = event_list.dim() == 3
        
        x = self.phi(event_list)
        
        if is_batched:
            # Batched processing: x is (Batch, N, 256)
            weights = 1.0 / (torch.norm(x, dim=2, keepdim=True) + 1e-6)
            # Sum across the sequence length (dim=1) for permutation invariance
            x = torch.sum(x * weights, dim=1) 
        else:
            # Single item processing: x is (N, 256)
            weights = 1.0 / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
            x = torch.sum(x * weights, dim=0, keepdim=True) 
            
        z_t = self.rho(x)
        return z_t