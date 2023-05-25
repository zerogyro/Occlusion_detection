import numpy as np
import sys
import os
import argparse
import math
from sklearn.cluster import KMeans
import matplotlib.image as mpimg


from utils import get_transition_matrix, convert_polar, convert_rawpcd_polar
from utils_ import loader


class Kernel_tool_unordered(object):
    """
    File discription
    This tool is for extracting keypoints from unordered depth map points
    """

    def __init__(self, pcd, cam, png):
        IMG_W, IMG_H, _ = png.shape
        self.gate_size = -1.2
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.pcd = pcd
        self.cam = cam
        self.polar_pcd = convert_rawpcd_polar(self.pcd)
        self.std_threshold = 3
        self.kernel_size = 4
        self.post_kernel_size = 5
        self.np_loc = None
        self.key_u, self.key_v, self.out = self.kernel_method(debug=False)

        # keypoints have shape of (n, 3) : n, number of points, 3 , u,v,depth
        self.keypoints = np.stack([self.key_u, self.key_v, self.out], axis=1)

    def get_mapping_dict(self):
        """
        this function generate two mapping dictionaries from pcd2img and img2pcd
        input: new_velo, new_cam from preprocesing mapped points from point cloud data and image frame
        output: cam2pc_dict, pc2cam_dict
        """
        new_cam, new_velo = self.cam, self.pcd
        u, v, d = new_cam
        x, y, z, k = new_velo
        l = len(u)
        cam2pc_dict = {}
        pc2cam_dict = {}
        for i in range(l):
            cam2pc_dict[u[i], v[i]] = [x[i], y[i], z[i]]
            pc2cam_dict[x[i], y[i], z[i]] = [u[i], v[i]]
        return cam2pc_dict, pc2cam_dict

    def sorting_frame(self, new_velo, new_cam):
        """
        sort new_velo and new_cam according to the pixel location of the frame
        input: new_velo, new_cam after mapping
        output: new_velo, new_cam after sorting
        """

        new_cam_t = new_cam.T
        new_velo_t = new_velo.T

        new_cam_t = np.array(new_cam_t)

        # a = np.array([(3, 2), (6, 2), (3, 6), (3, 4), (5, 3)])

        ind = np.lexsort((new_cam_t[:, 1], new_cam_t[:, 0]))

        res_new_cam_T = new_cam_t[ind]
        res_new_velo_T = new_velo_t[ind]

        new_cam, new_velo = res_new_cam_T.T, res_new_velo_T.T
        return new_velo, new_cam

    def gate(self, loc, cam2pc_dict, gate_size):
        cam2pc_key = (loc[0], loc[1])
        pc_v = cam2pc_dict[cam2pc_key]
        pc_z = pc_v[2]
        # print(pc_z,'----------------------------------')
        if pc_z < gate_size:
            return False
        return True

    def get_range_from_loc(self, cam, loc, size, std_thresh, out, key_u, key_v):
        # out = []
        u, v, d = cam
        # x,y,z,k = c
        # print(b.shape, c.shape)
        mid_u, mid_v = loc

        u_in = np.logical_and(u > mid_u - size, u < mid_u + size)
        v_in = np.logical_and(v > mid_v - size, v < mid_v + size)
        inlier = np.logical_and(u_in, v_in)

        v = np.std(d[inlier])
        if v > std_thresh:
            # print(loc)
            # print("std:", v)
            key_u.append(loc[0])
            key_v.append(loc[1])
            out.append(v)

    def filtering(self, new_cam, cam2pc_dict):
        new_cam_T = new_cam.T  # shape of (n,3)

        out = []
        key_u = []
        key_v = []

        for i in new_cam_T:
            loc = i[:2]
            if self.gate(loc, cam2pc_dict, self.gate_size):
                self.get_range_from_loc(
                    new_cam,
                    loc,
                    self.kernel_size,
                    self.std_threshold,
                    out,
                    key_u,
                    key_v,
                )
            # break

        return key_u, key_v, out

    def kernel_method(self, debug=False):
        IMG_H, IMG_W = self.IMG_H, self.IMG_W
        clean = np.zeros((IMG_H, IMG_W, 3))
        new_velo, new_cam = self.sorting_frame(self.pcd, self.cam)

        cam2pc_dict, pc2cam_dict = self.get_mapping_dict()

        key_u, key_v, out = self.filtering(new_cam, cam2pc_dict)

        return key_u, key_v, out


def test():
    file_name = "0000000000"
    png, velo = loader(file_name)
    from utils_ import get_mapped_points

    new_velo, new_cam = get_mapped_points(png, velo)
    # initialize the keypoint extractor for unordered points
    tool_a = Kernel_tool_unordered(new_velo, new_cam, png)

    keypoints = tool_a.keypoints

    for keypoint in keypoints:
        print(keypoint)


if __name__ == "__main__":
    test()
