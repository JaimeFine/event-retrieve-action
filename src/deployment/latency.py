import torch
import time
from agents import EventCentricAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Safety settings:
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

def benchmark(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()

    sync()
    t0 = time.perf_counter()

    for _ in range(iters):
        fn()

    sync()
    t1 = time.perf_counter()

    return (t1 - t0) / iters * 1000     # In ms

def hardware_audit_fp32(model, kb_data, sample):
    # 1. Load the snapshot into the model's memory object
    model.memory.latents = kb_data['latents'].to(device)
    model.memory.actions = kb_data['actions'].to(device)
    model.memory.reliability = kb_data['reliability'].to(device)
    
    model.eval().to(device)
    x = sample.to(device)

    with torch.inference_mode():
        # Baseline End-to-End
        total = benchmark(lambda: model(x))

        # Component 1: Encoder
        enc = benchmark(lambda: model.encoder(x))
        z = model.encoder(x)

        # Component 2: Knowledge Bank Retrieval
        ret = benchmark(lambda: model.memory.retrieve(z))
        weights, actions, _, _ = model.memory.retrieve(z)

        # Component 3: Vectorized Stability Check (The Lyapunov Block)
        def stable_block():
            # Testing the transition: z_next = z*Psi + a*Gamma
            z_next = z @ model.Psi + actions @ model.Gamma.t()
            return model.stabilizer.is_stable_batch(z, z_next)
        stab = benchmark(stable_block)

        # Component 4: Clustered Bayesian Selection (The Fusion Block)
        fuse = benchmark(
            lambda: model.clustered_bayesian_selection(
                actions, torch.ones(actions.shape[0], device=device)
            )
        )
        
        print("\n=== JETSON ORIN NANO AUDIT (FP32) ===")
        print(f"End-to-End Latency: {total:.3f} ms")
        print(f"Encoder Pass     : {enc:.3f} ms")
        print(f"KB Retrieval     : {ret:.3f} ms")
        print(f"Stability Check  : {stab:.3f} ms")
        print(f"Action Fusion    : {fuse:.3f} ms")

def hardware_audit_fp16(model, kb_data, sample):
    # 1. Load the snapshot into the model's memory object
    model.memory.latents = kb_data['latents'].to(device)
    model.memory.actions = kb_data['actions'].to(device)
    model.memory.reliability = kb_data['reliability'].to(device)
    
    model.eval().to(device)
    x = sample.to(device)

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        # Baseline End-to-End
        total = benchmark(lambda: model(x))

        # Component 1: Encoder
        enc = benchmark(lambda: model.encoder(x))
        z = model.encoder(x)

        # Component 2: Knowledge Bank Retrieval
        ret = benchmark(lambda: model.memory.retrieve(z))
        weights, actions, _, _ = model.memory.retrieve(z)

        # Component 3: Vectorized Stability Check (The Lyapunov Block)
        def stable_block():
            # Testing the transition: z_next = z*Psi + a*Gamma
            z_next = z @ model.Psi + actions @ model.Gamma.t()
            return model.stabilizer.is_stable_batch(z, z_next)
        stab = benchmark(stable_block)

        # Component 4: Clustered Bayesian Selection (The Fusion Block)
        fuse = benchmark(
            lambda: model.clustered_bayesian_selection(
                actions, torch.ones(actions.shape[0], device=device)
            )
        )
        
        print("\n=== JETSON ORIN NANO AUDIT (FP16) ===")
        print(f"End-to-End Latency: {total:.3f} ms")
        print(f"Encoder Pass     : {enc:.3f} ms")
        print(f"KB Retrieval     : {ret:.3f} ms")
        print(f"Stability Check  : {stab:.3f} ms")
        print(f"Action Fusion    : {fuse:.3f} ms")

model = EventCentricAgent()

weights = torch.load('agent_finetuned.pt', map_location='cuda')
model.encoder.load_state_dict(weights["encoder"])
model.Psi.data.copy_(weights["Psi"])
model.Gamma.data.copy_(weights["Gamma"])

kb = torch.load('knowledge_bank_snapshot.pt', map_location='cuda')

hardware_audit_fp32(model, kb, torch.randn(1, 13))
hardware_audit_fp16(model, kb, torch.randn(1, 13))
