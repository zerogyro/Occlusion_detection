import numpy as np
import os
import argparse
import math
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import time


from sensor_fusion_utils import get_mapped_points, get_sparse_depthmap, get_cam2velo_int

bin_path = "../test_data/0000000000.bin"
img_path = "../test_data/0000000000.png"

new_velo, new_cam = get_mapped_points(bin_path, img_path)
s_dmap = get_sparse_depthmap(new_cam)
cam2velo_dict = get_cam2velo_int(new_velo, new_cam)


IMG_H = 375
IMG_W = 1242


# dict has key with window(r_s, c_s)for upper left point and values for standard deviation
def general_kernel_process(s_dmap):
    # sliding window through whole sparse map
    # window_h, window_w, std_thresh

    img_h, img_w = s_dmap.shape
    assert img_h == IMG_H and img_w == IMG_W

    window_h = 5
    window_w = 5
    std_threshold = 2

    res_keypoint = {}
    for r_s in range(0, img_h + 1 - window_h):
        for c_s in range(0, img_w + 1 - window_w):
            # retriving points within window
            window = s_dmap[r_s : r_s + window_h, c_s : c_s + window_w]
            # get !0 depth
            new_window = window[np.where(window != 0)]

            if new_window.any():
                # getting keypoints
                # std:
                # std = sqrt(mean(x)), where x = abs(a - a.mean())**2.
                std_index = np.std(new_window)

                if std_index > std_threshold:
                    # print(new_window)
                    res_keypoint[(r_s, c_s)] = std_index
    return res_keypoint


# kernel_process_a returns dict with keypoint (u,v ) as keys and depth as values
# key: u,v in depth map for keypoint, value: std_
def get_keydict_std(s_dmap):
    img_h, img_w = s_dmap.shape
    assert img_h == IMG_H and img_w == IMG_W

    window_h = 5
    window_w = 5
    std_threshold = 2

    # keypoints in every u,v
    res_keypoint = {}
    for r_s in range(0, img_h + 1 - window_h):
        for c_s in range(0, img_w + 1 - window_w):
            # retriving points within window
            window = s_dmap[r_s : r_s + window_h, c_s : c_s + window_w]
            # get !0 depth
            uv_index = np.where(window != 0)
            new_window = window[uv_index]

            if new_window.any():
                # getting keypoints
                # std:
                # std = sqrt(mean(x)), where x = abs(a - a.mean())**2.
                std_index = np.std(new_window)

                if std_index > std_threshold:
                    np_uv_index = np.array(uv_index).T
                    for pair in np_uv_index:
                        u = r_s + pair[0]
                        v = c_s + pair[1]
                        res_keypoint[(u, v)] = s_dmap[u, v]

    return res_keypoint


# output: keypoints (u,v) in depth map
def get_keypoints_d(s_dmap):
    img_h, img_w = s_dmap.shape
    assert img_h == IMG_H and img_w == IMG_W

    window_h = 3
    window_w = 3
    std_threshold = 2

    # keypoints in every u,v
    res_keypoints = []
    for r_s in range(0, img_h + 1 - window_h):
        for c_s in range(0, img_w + 1 - window_w):
            # retriving points within window
            window = s_dmap[r_s : r_s + window_h, c_s : c_s + window_w]
            # get !0 depth
            uv_index = np.where(window != 0)
            new_window = window[uv_index]

            if new_window.any():
                # getting keypoints
                # std:
                # std = sqrt(mean(x)), where x = abs(a - a.mean())**2.
                std_index = np.std(new_window)

                if std_index > std_threshold:
                    np_uv_index = np.array(uv_index).T
                    for pair in np_uv_index:
                        u = r_s + pair[0]
                        v = c_s + pair[1]
                        res_keypoints.append([u, v])

    return res_keypoints


# keypoints_d = get_keypoints_d(s_dmap)
# keypoints_d = np.array(keypoints_d)
# print(keypoints_d.shape)


from utils import vis_keypoints_in_depth


# for key in res_keypoints.keys():
#     value_velo = cam2velo_dict[key]
#     print(key, value_velo)
