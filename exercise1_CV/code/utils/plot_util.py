import numpy as np


def plot_keypoints(ax, keypoints, weights, img_size=None,
                   draw_limbs=True, draw_kp=False):
    LIMBS_COCO = [[0, 2], [2, 4], [0, 1], [1, 3],  # head
                  [4, 6], [6, 8], [8, 10],  # right arm
                  [3, 5], [5, 7], [7, 9],  # left arm
                  [6, 12], [12, 14], [14, 16],  # right torso/leg
                  [5, 11], [11, 13], [13, 15],  # left torso/leg
                  [12, 11], [6, 5]  # torso horizontals
                  ]
    red, green, blue, cyan = (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0), (0, 1.0, 1.0)
    COLORS = [cyan, cyan, cyan, cyan,
              green, green, green,
              red, red, red,
              blue, green, green,
              blue, red, red,
              blue, blue
              ]

    keypoints = np.array(keypoints).reshape([17, 2])
    if img_size is not None:
        keypoints *= np.array([[img_size[1], img_size[0]]])

    m = np.array(weights) > 0.5
    if draw_limbs:
        for c, (p0, p1) in zip(COLORS, LIMBS_COCO):
            if m[p0] and m[p1]:
                ax.plot(keypoints[[p0, p1], 0], keypoints[[p0, p1], 1], color=c, linewidth=2)

    if draw_kp:
        ax.plot(keypoints[m, 0], keypoints[m, 1], 'ro')


def draw_keypoints(img, keypoints, weights, img_size=None):
    import cv2
    LIMBS_COCO = [[0, 2], [2, 4], [0, 1], [1, 3],  # head
                  [4, 6], [6, 8], [8, 10],  # right arm
                  [3, 5], [5, 7], [7, 9],  # left arm
                  [6, 12], [12, 14], [14, 16],  # right torso/leg
                  [5, 11], [11, 13], [13, 15],  # left torso/leg
                  [12, 11], [6, 5]  # torso horizontals
                  ]
    red, green, blue, cyan = (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)
    COLORS = [cyan, cyan, cyan, cyan,
              green, green, green,
              red, red, red,
              blue, green, green,
              blue, red, red,
              blue, blue
              ]

    keypoints = np.array(keypoints).reshape([17, 2])
    if img_size is not None:
        keypoints *= np.array([[img_size[1], img_size[0]]])

    m = np.array(weights) > 0.5
    for c, (p0, p1) in zip(COLORS, LIMBS_COCO):
        if m[p0] and m[p1]:
            pt1 = keypoints[p0, :].round()
            pt2 = keypoints[p1, :].round()
            img = cv2.line(img,
                           (int(pt1[0]), int(pt1[1])),
                           (int(pt2[0]), int(pt2[1])),
                           color=c, thickness=2)
    return img
