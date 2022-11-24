import numpy as np
from numba import jit
from utils import create_gaussian, add_noise
from scipy.ndimage import gaussian_filter
from tqdm import trange


@jit()
def weighted_dist(i, j, w, var=0.1):
    """
    This function is used to calculate the weighted distance between 2 patchs
    i is one patch and j is another, w is the gaussian kernel
    """
    diff = i - j
    return (w*(diff*diff - 2*var)).sum()

@jit()
def nl_means(gt_image, patch_size, search_size, h, sigma, a, noise_mode="gaussian"):
    w = create_gaussian((patch_size, patch_size), a)

    # padding images so that corner pixels get proper justice- 
    # zero padding would make it difficult for corner pixels to find similar patches
    to_pad = patch_size//2
    noisy_image = add_noise(gt_image, noise_mode)
    padded_noisy_image = np.pad(noisy_image, (to_pad, to_pad), mode="reflect")
    
    # initializing the output
    pred=np.zeros(gt_image.shape)
    
    # first 2 for loops to get the central pixel of the patch under consideration
    for i in trange(to_pad, padded_noisy_image.shape[0]-to_pad, desc="Running NLmeans"):
        for j in range(to_pad, padded_noisy_image.shape[1]-to_pad):
            curr_patch = padded_noisy_image[i-to_pad:i+to_pad+1, j-to_pad:j+to_pad+1]

            # this is just to sum all the weights (as Z(i) does)
            total_sum = 0

            # going thorough search window
            for a in range(i-search_size//2, i+(search_size//2)+1):
                
                # removing corner cases
                if a-to_pad<0:
                    continue
                if a+to_pad>=padded_noisy_image.shape[0]:
                    break

                for b in range(j-search_size//2, j+(search_size//2)+1):
                    if b-to_pad<0 or (a==i and b==j):
                        continue
                    if b+to_pad>=padded_noisy_image.shape[1]:
                        break
                    
                    search_patch = padded_noisy_image[a-to_pad:a+to_pad+1, b-to_pad:b+to_pad+1]
                    
                    # finding weights for search window patch and current patch
                    weight = np.exp(-weighted_dist(curr_patch, search_patch, w, sigma**2)/h**2)
                
                    total_sum += weight

                    # directly adding all the weighted patches. normalizing in end
                    pred[i-to_pad, j-to_pad] += weight*padded_noisy_image[a, b]

            # normalizing the pixel output
            pred[i-to_pad, j-to_pad] /=total_sum
    gauss_out = gaussian_filter(noisy_image, 0.75)
    return pred, gauss_out, noisy_image 
