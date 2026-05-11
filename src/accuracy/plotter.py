import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 15,
})

# --------------------------------------------------
# Load
# --------------------------------------------------
audit = torch.load("audit_tensors.pt", map_location="cpu")
ckpt  = torch.load("agent_finetuned.pt", map_location="cpu")

Psi   = ckpt["Psi"].float()
Gamma = ckpt["Gamma"].float()

z_t    = torch.tensor(np.vstack(audit["z_t"])).float()
a_t    = torch.tensor(np.vstack(audit["a_t"])).float()
a_exp  = torch.tensor(np.vstack(audit["a_exp"])).float()
z_next = torch.tensor(np.vstack(audit["z_next"])).float()

weights   = np.array(audit["weights"])
min_dist  = np.array(audit["min_dist"])
goal_dist = np.array(audit["goal_dist"])

T = len(z_t)
time = np.arange(T)

print("Frames:", T)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

# Transition model
z_pred = z_t @ Psi + a_t @ Gamma.t()

transition_error = F.mse_loss(
    z_pred, z_next, reduction="none"
).mean(dim=1).numpy()

# Alignment
cosine = F.cosine_similarity(a_t, a_exp, dim=1).numpy()

# Energy
V = torch.norm(z_t, dim=1).pow(2).numpy()
dV = np.diff(V, prepend=V[0])

# Retrieval confidence
conf = weights.max(axis=1)

# --------------------------------------------------
# Smoothing
# --------------------------------------------------
def smooth(x, w=7):
    box = np.ones(w)/w
    return np.convolve(x, box, mode="same")

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, axs = plt.subplots(2,1, figsize=(10,10))

# A ----------------------------------------------
ax = axs[0]
ax.plot(time, transition_error, alpha=0.3, label="Transition Error")
ax.plot(
    time, smooth(transition_error), lw=2, \
        label="Transition Error (Smooth)"
)
ax.set_title("A. Latent Transition Fidelity")
ax.set_ylabel("MSE")
ax.set_xlabel("Times")
ax.legend()
ax.grid(alpha=0.3)

# B ----------------------------------------------
ax = axs[1]
ax.plot(time, dV, alpha=0.5, label="Stability")
ax.axhline(0, ls="--", c="black")
ax.plot(time, smooth(dV), lw=2, label="Stability (Smooth)")
ax.set_title("C. Contractive Stability ($\Delta V_t$)")
ax.set_ylabel("$V(z_{t+1})-V(z_t)$")
ax.set_xlabel("Times")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figure_audit_results.pdf", dpi=300)
print("Saved figure_audit_results.pdf")