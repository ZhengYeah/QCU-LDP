import numpy as np

def merge_dim_of_2d_img(img, twice_grid_step=7):
    """
    Merge the two dimensions of a 2D image to form a 1D image
    :param img: (ndarray) a 2D image
    :param twice_grid_step: (int) the step size of the grid
    :return: (list of 2D index) indexes of the low-dimensional representation dims: merged and unmerged dims
    """
    assert isinstance(img, np.ndarray), "The input image must be a numpy array"
    assert img.ndim == 2, "The input image must be a 2D image"
    assert img.shape[0] == img.shape[1], "The input image must be a square image"
    assert img.shape[0] % twice_grid_step == 0, "The input image size must be divisible by twice_grid_step"

    merged_indexes = []
    unmerged_indexes = []
    for i in range(0, img.shape[0], twice_grid_step):
        for j in range(0, img.shape[1], twice_grid_step):
            block = img[i:i + twice_grid_step, j:j + twice_grid_step]
            if np.all(block == block[0, 0]):
                merged_indexes.append((i, j))
                continue
            else:
                for k in range(twice_grid_step):
                    for l in range(twice_grid_step):
                        unmerged_indexes.append((i + k, j + l))
    return merged_indexes, unmerged_indexes, twice_grid_step
