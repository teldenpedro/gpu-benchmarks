import cupy as cp
import numpy as np
from PIL import Image
from utils.timer import Timer
from utils.logger import log
import os

def load_image(path):
    img = Image.open(path).convert("L")  # convert to grayscale
    return np.array(img, dtype=np.float32)

def save_image(array, path):
    img = Image.fromarray(np.uint8(array))
    img.save(path)

def apply_filter_gpu(image_np, kernel_np):
    image_cp = cp.asarray(image_np)
    kernel_cp = cp.asarray(kernel_np)

    # Get image and kernel dimensions
    h, w = image_cp.shape
    kh, kw = kernel_cp.shape

    # Pad image
    pad_h, pad_w = kh // 2, kw // 2
    padded = cp.pad(image_cp, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    output = cp.zeros_like(image_cp)

    # Perform convolution manually
    for i in range(kh):
        for j in range(kw):
            output += kernel_cp[i, j] * padded[i:i+h, j:j+w]

    return cp.asnumpy(output)

def image_filtering():
    log("Loading image...")
    image = load_image("input_image.jpg")  # Make sure the image is in the project root

    # Define a sharpening filter (3x3)
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    log("Applying 3x3 filter using GPU...")

    with Timer() as t:
        result = apply_filter_gpu(image, kernel)

    log(f"Filtering completed in {t.interval:.6f} seconds.")

    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)

    # Save the filtered image
    save_image(result, "results/blurred_output.jpg")
    log("Saved result image as 'results/blurred_output.jpg'.")

    # Save the benchmark result
    with open("results/image_filter_result.txt", "w") as f:
        f.write("Filter applied: Sharpening (3x3)\n")
        f.write(f"Time taken: {t.interval:.6f} seconds\n")

if __name__ == "__main__":
    image_filtering()
