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
def add_noise_salt_and_pepper_noise(image, amount = 0.05):
    """
    This function is used to add salt and pepper noise of given image
    """
    # Converting pixel values from 0-255 to 0-1 float
    img = image / 255

    # Getting the dimensions of the image
    h = img.shape[0]
    w = img.shape[1]

    # Setting the ratio of salt and pepper in salt and pepper noised image
    s = 0.5
    p = 0.5

    # Initializing the result (noisy) image
    result = img.copy()

    # Adding salt noise to the image
    salt = np.ceil(amount * img.size * s)
    vec = []
    for i in img.shape:
        vec.append(np.random.randint(0, i - 1, int(salt)))

    result[vec] = 1

    # Adding pepper noise to the image
    pepper = np.ceil(amount * img.size * p)
    vec = []
    for i in img.shape:
        vec.append(np.random.randint(0, i - 1, int(pepper)))

    result[vec] = 0

    # Converting the result back to uint8
    result = np.uint8(result * 255)

    return result


def add_noise_gaussian(image, mean=0,var=0.01):
    """
    This function is used to add gaussian noise to given image
    """
    img = image / 255

    # Initializing the result (noisy) image
    result = img.copy()

    # Adding gaussian noise to the image
    gauss = np.random.normal(mean, var ** 0.5, img.shape)
    result = result + gauss
    result = np.clip(result, 0, 1)

    # Converting the result back to uint8
    result = np.uint8(result * 255)

    return result

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


def MSE(image1, image2):
    """
    Mean Squared Error
    :param image1: image1
    :param image2: image2
    :rtype: float
    :return: MSE value
    """

    # Calculating the Mean Squared Error
    mse = np.mean(np.square(image1.astype(np.float) - image2.astype(np.float)))

    return mse


def PSNR(image1, image2, peak=255):
    """
    Peak signal-to-noise ratio
    :param image1: image1
    :param image2: image2
    :param peak: max value of pixel 8-bit image (255)
    :rtype: float
    :return: PSNR value
    """

    # Calculating the Mean Squared Error
    mse = MSE(image1, image2)

    # Calculating the Peak Signal Noise Ratio
    psnr = 10 * np.log10(peak ** 2 / mse)

    return psnr