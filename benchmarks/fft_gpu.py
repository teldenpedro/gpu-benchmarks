import cupy as cp
from utils.timer import Timer
from utils.logger import log
import os

def fft_gpu():
    n = 2**20  # 1 million points
    x = cp.random.rand(n).astype(cp.complex64)

    log(f"Running FFT on {n} elements using GPU...")
    with Timer() as t:
        y = cp.fft.fft(x)
    log(f"FFT completed in {t.interval:.6f} seconds.")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/fft_result.txt", "w") as f:
        f.write(f"Input size: {n}\n")
        f.write(f"FFT completed in {t.interval:.6f} seconds\n")

if __name__ == "__main__":
    fft_gpu()
