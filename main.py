import os
from nl_means import nl_means
from utils import load_images, save_all, print_metrics
import yaml
import warnings

if __name__ == "__main__":
    # assumption for the various parameters
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    noise_type = config["noise_type"]
    patch_size = config["patch_size"]
    search_size = config["search_size"]
    h = config["h"]
    sigma = config["sigma"]
    a = config["a"]
    img_folder, img_name = config["img_folder"], config["img_name"]
    PATH = os.path.join(img_folder, img_name)
    save_folder = config["save_folder"]
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    image = load_images(PATH)
    warnings.filterwarnings("ignore")

    # Denoising
    nl_out, gauss_out, noisy_image = nl_means(image, patch_size, search_size, h, sigma, a, noise_type)

    save_name = os.path.join(save_folder, f"{noise_type}_{img_name}")
    save_all(noisy_image, gauss_out, nl_out, save_name)

    print_metrics(image, noisy_image, gauss_out, nl_out)
