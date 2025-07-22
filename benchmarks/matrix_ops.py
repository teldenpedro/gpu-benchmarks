import cupy as cp
from utils.timer import Timer
from utils.logger import log
import os

def matrix_multiplication():
    n = 1024
    a = cp.random.rand(n, n).astype(cp.float32)
    b = cp.random.rand(n, n).astype(cp.float32)

    log(f"Multiplying {n}x{n} matrices on GPU...")
    with Timer() as t:
        c = cp.dot(a, b)
    log(f"Done. Time taken: {t.interval:.6f} seconds")

    # Save result
    os.makedirs("results", exist_ok=True)
    with open("results/matrix_result.txt", "w") as f:
        f.write(f"Matrix size: {n}x{n}\n")
        f.write(f"Multiplication time: {t.interval:.6f} seconds\n")

if __name__ == "__main__":
    matrix_multiplication()
