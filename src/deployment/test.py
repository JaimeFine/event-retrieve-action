import torch
import os

def estimate_footprint(model_path, kb_path):
    model_size = os.path.getsize(model_path) / (1024**2)
    kb = torch.load(kb_path, map_location='cpu')

    kb_bytes = 0

    def accumulate_tensors(obj):
        nonlocal kb_bytes

        if isinstance(obj, torch.Tensor):
            kb_bytes += obj.nelement() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                accumulate_tensors(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                accumulate_tensors(v)

    accumulate_tensors(kb)
    kb_size = kb_bytes / (1024**2)

    total = model_size + kb_size

    print(f"--- ERA Deployment Memory Audit ---")
    print(f"Model (Encoder + Ψ + Γ): {model_size:.2f} MB")
    print(f"Knowledge Bank (Event Memory): {kb_size:.2f} MB")
    print(f"Total Static Load: {total:.2f} MB")
    print(f"Safety Margin on 4GB: {4096 - (total + 1200):.2f} MB")  # OS reserve

estimate_footprint(
    'agent_finetuned.pt',
    'knowledge_bank_snapshot.pt'
)

"""
--- ERA Deployment Memory Audit ---
Model (Encoder + Ψ + Γ): 0.33 MB
Knowledge Bank (Event Memory): 15.43 MB
Total Static Load: 15.77 MB
Safety Margin on 4GB: 2880.23 MB
"""