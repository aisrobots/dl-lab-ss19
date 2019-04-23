import matplotlib.pyplot as plt
import numpy as np

from model.data_seg import get_data_loader
from utils.plot_util import plot_keypoints


if __name__ == '__main__':
    """
        Script to show samples of the dataset
    """
    reader = get_data_loader()

    for idx, (img, msk) in enumerate(reader):
        print('img', type(img), img.shape)
        print('msk', type(msk), msk.shape)

        # turn image tensor into numpy array containing correctly scaled RGB image
        img_rgb = ((np.array(img) + 1.0)*127.5).round().astype(np.uint8).transpose([0, 2, 3, 1])

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img_rgb[0]); ax1.axis('off')
        ax2.imshow(msk[0]); ax2.axis('off')
        plt.show()
