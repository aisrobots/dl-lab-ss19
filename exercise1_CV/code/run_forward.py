import torch
import numpy as np
import matplotlib.pyplot as plt

from model.model import ResNetModel
from model.data import get_data_loader
from utils.plot_util import plot_keypoints


def normalize_keypoints(keypoints, img_shape):
    if img_shape[-1] != img_shape[-2]:
        raise ValueError("Only square images are supported")
    return keypoints/img_shape[-1]


if __name__ == '__main__':
    PATH_TO_CKPT = './trained_net.model'

    # create device and model
    cuda = torch.device('cuda')
    model = ResNetModel(pretrained=True)
    model.load_state_dict(torch.load(PATH_TO_CKPT))
    model.to(cuda)

    val_loader = get_data_loader(batch_size=1,
                                 is_train=False)

    for idx, (img, keypoints, weights) in enumerate(val_loader):
        img = img.to(cuda)
        keypoints = keypoints.to(cuda)
        weights = weights.to(cuda)

        # normalize keypoints to [0, 1] range
        keypoints = normalize_keypoints(keypoints, img.shape)

        # apply model
        pred = model(img, '')

        # show results
        img_np = np.transpose(img.cpu().detach().numpy(), [0, 2, 3, 1])
        img_np = np.round((img_np + 1.0) * 127.5).astype(np.uint8)
        kp_pred = pred.cpu().detach().numpy().reshape([-1, 17, 2])
        kp_gt = keypoints.cpu().detach().numpy().reshape([-1, 17, 2])
        vis = weights.cpu().detach().numpy().reshape([-1, 17])

        for bid in range(img_np.shape[0]):
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(img_np[bid]), ax1.axis('off'), ax1.set_title('input + gt')
            plot_keypoints(ax1, kp_gt[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
            ax2.imshow(img_np[bid]), ax2.axis('off'), ax2.set_title('input + pred')
            plot_keypoints(ax2, kp_pred[bid], vis[bid], img_size=img_np[bid].shape[:2], draw_limbs=True, draw_kp=True)
            plt.show()
