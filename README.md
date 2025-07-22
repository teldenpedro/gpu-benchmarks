Project by-Pragneya Joshi 
# GPU Benchmarks with CuPy

This project demonstrates GPU-accelerated benchmarks using CuPy, a NumPy-compatible library for NVIDIA GPUs. It includes matrix operations, image filtering, FFT, and sorting — all running on CUDA.

## Project Structure
- `benchmarks/`: Contains benchmark scripts
  - `matrix_ops.py`: Matrix multiplication
  - `image_filters.py`: 2D image filtering
  - `sorting_algos.py`: Sorting large arrays
  - `fft_gpu.py`: Fast Fourier Transform
- `utils/`: Helper modules
  - `timer.py`: Benchmark timer
  - `logger.py`: Console logger
- `requirements.txt`: Dependencies
- `README.md`: Project overview

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Running-
python -m benchmarks.matrix_ops
python -m benchmarks.image_filters
python -m benchmarks.sorting_algos
python -m benchmarks.fft_gpu
