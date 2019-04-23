import matplotlib.pyplot as plt
import numpy as np

from model.data import get_data_loader
from utils.plot_util import plot_keypoints


if __name__ == '__main__':
    """
        Script to show samples of the dataset
    """
    reader = get_data_loader()

    for idx, (img, keypoints, weights) in enumerate(reader):
        print('img', type(img), img.shape)
        print('keypoints', type(keypoints), keypoints.shape)
        print('weights', type(weights), weights.shape)

        # turn image tensor into numpy array containing correctly scaled RGB image
        img_rgb = ((np.array(img) + 1.0)*127.5).round().astype(np.uint8).transpose([0, 2, 3, 1])

        # show
        plt.figure()
        plt.imshow(img_rgb[0]); plt.axis('off'); ax = plt.gca()
        plot_keypoints(ax, keypoints[0], weights[0], draw_limbs=True, draw_kp=True)
        plt.show()
