import torch
import time

def benchmark(device, n, repeats=30):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # Warm-up
    for _ in range(5):
        c = a @ b

    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    for _ in range(repeats):
        c = a @ b

    if device == "mps":
        torch.mps.synchronize()

    return (time.perf_counter() - start) / repeats * 1000

for n in [32, 128, 512, 1024, 2048]:
    cpu_ms = benchmark("cpu", n)
    gpu_ms = benchmark("mps", n)
    print(f"{n=:5d} CPU={cpu_ms:.3f}ms MPS={gpu_ms:.3f}ms")
