import cupy as cp
from utils.timer import Timer
from utils.logger import log
import os

def gpu_sort():
    n = 10_000_000
    data = cp.random.rand(n).astype(cp.float32)

    log(f"Sorting array of {n:,} elements on GPU...")
    with Timer() as t:
        sorted_data = cp.sort(data)
    log(f"Sorting completed in {t.interval:.6f} seconds.")

    # Save result
    os.makedirs("results", exist_ok=True)
    with open("results/sort_result.txt", "w") as f:
        f.write(f"Array size: {n}\n")
        f.write(f"Sorting time: {t.interval:.6f} seconds\n")

if __name__ == "__main__":
    gpu_sort()

