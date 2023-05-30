import numpy as np
import os
import argparse
import math
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import time


from sensor_fusion import get_mapped_points, get_sparse_depthmap, get_cam2velo_int

bin_path = "test_data/0000000000.bin"
img_path = "test_data/0000000000.png"

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
def kernel_process_a(s_dmap):
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


def post_process(list_keypoint, k):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(list_keypoint)
    return kmeans.inertia_
    # print(kmeans.cluster_centers_)


def get_cluster_centers(list_keypoint, k):
    keypoints = keypoint_dict.keys()
    np_keypoints = np.array(list(keypoints))
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(np_keypoints)
    return kmeans.cluster_centers_


def get_distortions(keypoint_dict, k_cluster=20):
    keypoints = keypoint_dict.keys()
    np_keypoints = np.array(list(keypoints))
    distortions = []
    for i in range(1, k_cluster + 1):
        start_time = time.time()
        index = post_process(np_keypoints, i)
        distortions.append(index)
        print(
            f"processing time {time.time() - start_time} with {i} clusters, and inertia_ = {index}"
        )

    return distortions


def get_elbow_k_clusters(distortions):
    def get_partial(distortions):
        d_ = []
        for i in range(len(distortions) - 1):
            derivative = distortions[i + 1] - distortions[i]
            d_.append(derivative)
        return d_

    d1_ = get_partial(distortions)
    d2_ = get_partial(d1_)

    return d1_, d2_


keypoint_dict = kernel_process_a(s_dmap)
distortions = get_distortions(keypoint_dict, k_cluster=20)
distortions = np.array(distortions)
distortions /= 1e6


d1_, d2_ = get_elbow_k_clusters(distortions)


cluster_centers_6 = get_cluster_centers(keypoint_dict, k=6)
print(cluster_centers_6)


# vis_utils for sparse_kernel
def show_img(img_path):
    png = mpimg.imread(img_path)
    return png
