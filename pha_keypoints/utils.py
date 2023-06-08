import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IMG_H = 375
IMG_W = 1242


def vis_depth_raw(s_dmap):
    where_0 = np.where(s_dmap == 0)
    vis_s_dmap = s_dmap / np.max(s_dmap)
    vis_s_dmap[where_0] = 255
    return vis_s_dmap


def vis_keypoints_in_depth(s_dmap, keypoints):
    where_0 = np.where(s_dmap == 0)

    d_canvas = np.ones((IMG_H, IMG_W, 3), np.uint8)
    d_canvas *= 255
    d_canvas[where_0] = [0, 0, 0]
    for key in keypoints:
        d_canvas[key[0], key[1]] = [0, 255, 255]
    res = d_canvas
    return res


def img_loader(path):
    png = mpimg.imread(path)
    return png


def vis_keypoints_on_rgb(img, keypoints):
    for key in keypoints:
        img[key[0]][key[1]] = [0, 255, 255]
    return img
