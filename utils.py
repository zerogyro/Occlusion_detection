import numpy as np
import os
import argparse
import matplotlib.image as mpimg
from matplotlib.cm import get_cmap


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


class Depth_converter(object):
    def __init__(self, args):
        # TODO: get_trainsition_matrix from args

        self.bin_path = os.path.join(args.test_folder, args.bin_path)
        self.img_path = os.path.join(args.test_folder, args.img_path)

        self.P2, self.R0_rect, self.Tr_velo_to_cam = get_transition_matrix()

    def get_mapped_points(self):
        """Mapping point cloud data to image data that points that left are the points in camera frame.
        Returns:
            new_velo: (4,n) x,y,z,i/number of points
            new_cam: (3,n) u,v,d / number of points
        """
        binary_path, img_path = self.bin_path, self.img_path
        png = mpimg.imread(img_path)
        IMG_H, IMG_W, _ = png.shape
        # read raw data from binary
        scan = np.fromfile(binary_path, dtype=np.float32).reshape((-1, 4))
        points = scan[:, 0:3]  # lidar xyz (front, left, up)
        # print(points[0])
        velo = np.insert(points, 3, 1, axis=1).T
        velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
        cam = self.P2 * self.R0_rect * self.Tr_velo_to_cam * velo

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


class Kernel_tool(object):
    def __init__(self, pcd, cam):
        self.gate_size = -1.2
        self.IMG_H = 375
        self.IMG_W = 1242
        self.pcd = pcd
        self.cam = cam
        self.std_threshold = 3
        self.kernel_size = 4

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
