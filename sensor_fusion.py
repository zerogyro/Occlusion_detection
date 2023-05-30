import numpy as np
import os
import argparse
import math
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
from matplotlib.cm import get_cmap
from scipy.interpolate import LinearNDInterpolator


def get_transition_matrix():
    """This function returns the transition matrix from calibration between Lidar and mono-camera

    Returns:
        numpy.matrix Transition Matrix

    """
    calib_file = "calib/calib.txt"
    with open(calib_file, "r") as f:
        calib = f.readlines()

    # P2 (3 x 4) for left eye
    P2 = np.matrix([float(x) for x in calib[2].strip("\n").split(" ")[1:]]).reshape(
        3, 4
    )
    R0_rect = np.matrix(
        [float(x) for x in calib[4].strip("\n").split(" ")[1:]]
    ).reshape(3, 3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.matrix(
        [float(x) for x in calib[5].strip("\n").split(" ")[1:]]
    ).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    return P2, R0_rect, Tr_velo_to_cam


P2, R0_rect, Tr_velo_to_cam = get_transition_matrix()


def get_mapped_points(bin_path, img_path):
    """Mapping point cloud data to image data that points that left are the points in camera frame.
    Returns:
        new_velo: (4,n) x,y,z,i/number of points
        new_cam: (3,n) u,v,d / number of points
    """
    binary_path, img_path = bin_path, img_path
    png = mpimg.imread(img_path)
    IMG_H, IMG_W, _ = png.shape
    # read raw data from binary
    scan = np.fromfile(binary_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]  # lidar xyz (front, left, up)
    # print(points[0])
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)

    cam = P2 * R0_rect * Tr_velo_to_cam * velo
    # 1 to 1 mapping for new_velo and cam
    cam_index = np.where(cam[2, :] >= 0)[1]
    new_velo = np.take(velo, cam_index, axis=1)
    cam = np.delete(cam, np.where(cam[2, :] < 0)[1], axis=1)
    # get u,v,z
    cam[:2] /= cam[2, :]
    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)

    new_velo = np.delete(new_velo, np.where(outlier), axis=1)
    new_cam = np.delete(cam, np.where(outlier), axis=1)
    new_cam = np.asarray(new_cam)
    return new_velo, new_cam


def get_cam2velo_int(new_velo, new_cam):
    xyz = new_velo[:3, :]
    xyz = xyz.T
    u, v, d = new_cam
    res_vel2cam_dict = {}
    u = u.astype(int)
    v = v.astype(int)
    uv = np.stack((u, v))
    uv_list = uv.T
    for i, pair in enumerate(uv_list):
        res_vel2cam_dict[(pair[0], pair[1])] = xyz[i]
    return res_vel2cam_dict


#####WARNING: SPARSE_DEPTHMAP SHAPE
def get_sparse_depthmap(new_cam):
    u, v, z = new_cam
    u = u.astype(int)
    v = v.astype(int)
    assert u.dtype == int
    uv = np.stack((u, v))
    uv_list = uv.T

    # init depth map
    sparse_depthmap = np.zeros((375, 1242))
    for i, pixel_loc in enumerate(uv_list):
        sparse_depthmap[pixel_loc[1], pixel_loc[0]] = z[i]

    return sparse_depthmap


if __name__ == "__main__":
    bin_path = "test_data/0000000000.bin"
    img_path = "test_data/0000000000.png"

    new_velo, new_cam = get_mapped_points(bin_path, img_path)
    s_dmap = get_sparse_depthmap(new_cam)
