import cv2
from numba import jit
import numpy as np
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from skimage.restoration import denoise_nl_means

import matplotlib.pyplot as plt


@jit()
def create_gaussian(shape=(7, 7), a=1.5):
    """
    This is used to create weights which is distributed as bivariate gaussian kernel
    """

    tmpx = (shape[0] - 1) / 2
    tmpy = (shape[1] - 1) / 2
    x_idx = np.arange(-tmpx, tmpx + 1)
    y_idx = np.arange(-tmpy, tmpy + 1)
    y_idx = y_idx.reshape(shape[1], 1)

    # bivariate gaussian
    gauss = np.exp(-(x_idx * x_idx + y_idx * y_idx) / (2 * a * a))

    # normalizing gaussian weights
    gauss = np.divide(gauss, np.sum(gauss))
    return gauss


@jit()
def add_noise(image, mode="gaussian"):
    """
    This function is used to add noise of given mode
    """
    noisy_image = random_noise(image, mode)
    return noisy_image


def load_images(PATH):
    image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    image = np.divide(image, 255)
    return image


def save_all(noisy_image, gauss_output, nl_output, save_name, noise_type="Gaussian"):
    # plotting Gaussian noisy, gaussian filtered and NL means denoised images together

    fig = plt.figure(figsize=(22, 14))
    plt.subplot(1, 3, 1)
    plt.title(f"After Adding {noise_type} Noise")
    plt.imshow(noisy_image, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Gaussian Filtering")
    plt.imshow(gauss_output, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Denoised Image")
    plt.axis("off")
    plt.imshow(nl_output, cmap="gray", vmin=0, vmax=1)
    plt.savefig(save_name)


def print_metrics(gt_image, noisy_image, guass_output, nl_output, noise_type="Gaussian"):
    print("-" * 100)
    print(" " * 30 + f"PSNR (all compared with true image) {noise_type} Noise")
    print("-" * 100)
    print(f"Noisy image = {peak_signal_noise_ratio(gt_image, noisy_image)}")
    print(f"Gaussian filtering = {peak_signal_noise_ratio(gt_image, guass_output)}")
    print(f"Non-Local Means = {peak_signal_noise_ratio(gt_image, nl_output)}")

    print("\n")
    print("-" * 100)
    print(" " * 50 + "MSE error")
    print("-" * 100)
    print(f"MSE error for noisy image = {mean_squared_error(gt_image, noisy_image)}")
    print(f"MSE error after gaussian filter = {mean_squared_error(gt_image, guass_output)}")
    print(f"MSE error for predicted image = {mean_squared_error(gt_image, nl_output)}")
