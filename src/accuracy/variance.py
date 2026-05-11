import torch
import numpy as np

# Load the audit data
data = torch.load('audit_tensors.pt', map_location='cpu')
z_t = np.vstack(data['z_t'])

# Calculate the 'Global' variance of all latent features
# This represents the total "territory" the latent space covers
total_variance = np.var(z_t)

print(f"Latent Dimension: {z_t.shape[1]}")
print(f"Global Latent Variance: {total_variance:.4f}")
