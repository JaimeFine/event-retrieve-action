import torch
import gc
from agents import EventCentricAgent

def profile_memory_usage(model, kb_data, sample_input):
    device = torch.device('cuda')
    def get_vram():
        # In MB:
        return torch.cuda.memory_allocated(device) / 1024**2
    
    print(f"{'='*10} MEMORY AUDIT {'='*10}")

    # 1. Baseline (Empty GPU)
    gc.collect()
    torch.cuda.empty_cache()
    base_mem = get_vram()
    print(f"Base GPU Memory   : {base_mem:.2f} MB")

    # 2. Model Static Footprint (FP32)
    model.to(device)
    model_mem = get_vram() - base_mem
    print(f"Model (FP32)      : {model_mem:.2f} MB")

    # 3. Knowledge Bank Footprint
    kb_size = 0
    for key in kb_data:
        kb_size += kb_data[key].element_size() * kb_data[key].nelement()
    kb_size_mb = kb_size / 1024**2
    print(
        f"Knowledge Bank    : {kb_size_mb:.2f} MB "
        f"({len(kb_data['latents'])} samples)"
    )

    # 4. Peak Inference Memory
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _ = model(sample_input.to(device))

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Peak Inf. (FP32)  : {peak_mem:.2f} MB")

    # 5. Quantization Impact (FP16)
    model.half()
    model_mem_half = get_vram() - base_mem

    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _ = model(sample_input.to(device).half())

    peak_mem_half = torch.cuda.max_memory_allocated(device) / 1024**2
    reduction = (1 - model_mem_half / (model_mem + 1e-9)) * 100
    print(
        f"\nModel (FP16)      : {model_mem_half:.2f} MB "
        f"({reduction:.1f}% reduction)"
    )
    print(f"Peak Inf. (FP16)  : {peak_mem_half:.2f} MB")
    
    total_estimated = model_mem_half + (kb_size_mb / 2) # Est. KB in FP16
    print(f"\nEstimated Total Footprint (FP16): {total_estimated:.2f} MB")
    print(f"Safety Margin on 4GB Jetson: {4096 - total_estimated:.2f} MB")

model = EventCentricAgent()

weights = torch.load('agent_finetuned.pt', map_location='cuda')
model.encoder.load_state_dict(weights["encoder"])
model.Psi.data.copy_(weights["Psi"])
model.Gamma.data.copy_(weights["Gamma"])

kb = torch.load('knowledge_bank_snapshot.pt', map_location='cuda')

profile_memory_usage(model, kb, torch.randn(1, 13))