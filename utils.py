import numpy as np
import os
import argparse
import math
from sklearn.cluster import KMeans
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


def convert_polar(np_pcd):
    np_xy = np.array(np_pcd[:, :-1], dtype=object)
    polar_index = []
    # print('start')
    check = []
    for pair in np_xy:
        r = math.sqrt(pair[0] ** 2 + pair[1] ** 2)
        theta = (math.atan2(pair[0], pair[1]) / math.pi) * 180
        polar_index.append([theta, r])
        # check.append([theta, r, pair[0], pair[1]])
    # print(polar_index)
    polar_index = np.array(polar_index)
    # check = np.array(check)
    return polar_index


def convert_rawpcd_polar(raw_pcd):
    """raw_pcd with shape (4, n)
        4 : x, y ,z i
        n : number of points
    Args:
        raw_pcd in  kernel_tool.pcd
    Returns:
        polar_pcd in numpy matrix
    """
    xyz_pcd = raw_pcd[:3, :]
    xyz_pcd = xyz_pcd.T

    ### xyz_pcd with shape (n,3)
    np_xy = np.array(xyz_pcd[:, :-1], dtype=object)
    polar_index = []
    # print('start')
    check = []
    for pair in np_xy:
        r = math.sqrt(pair[0] ** 2 + pair[1] ** 2)
        theta = (math.atan2(pair[0], pair[1]) / math.pi) * 180
        polar_index.append([theta, r])
        # check.append([theta, r, pair[0], pair[1]])
    # print(polar_index)
    polar_index = np.array(polar_index)
    # check = np.array(check)
    return polar_index


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
        self.polar_pcd = convert_rawpcd_polar(self.pcd)
        self.std_threshold = 1
        self.kernel_size = 5
        self.post_kernel_size = 5
        self.np_loc = None

        self.key_u, self.key_v, _ = self.kernel_method(debug=False)

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

    def get_kernel_dict(self, key_v, key_u):
        cam2pc_dict, _ = self.get_mapping_dict()
        np_key_v = np.array(key_v)
        np_key_u = np.array(key_u)
        key_len = len(np_key_v)
        assert key_len == len(np_key_u)
        np_loc = np.stack((np_key_u, np_key_v), axis=-1)
        self.np_loc = np_loc

        kernel_cam2pc_dict = {}
        for k in np_loc:
            # print(k)
            key = (k[0], k[1])
            v = cam2pc_dict[key]
            kernel_cam2pc_dict[key] = v
        # print(kernel_cam2pc_dict)
        return kernel_cam2pc_dict

    def post_kernel_method(self, loc):
        np_loc = self.np_loc
        size = self.post_kernel_size
        key_u, key_v = self.key_u, self.key_v
        # print('start kernel method')
        # first get kernel size
        # key_v
        # key_u
        # print(len(key_u),len(key_v))

        # print('preprocessing kernel method for kernel_uv, kernel_pcd, kernel_polar')
        mid_u, mid_v = loc
        # print(mid_u, mid_v)

        # print(key_u[0],key_v[0])
        u_in = np.logical_and(key_u > mid_u - size, key_u < mid_u + size)
        v_in = np.logical_and(key_v > mid_v - size, key_v < mid_v + size)
        inlier = np.logical_and(u_in, v_in)

        # print(inlier)

        kernel_uv = np_loc[inlier]
        kernel_pcd = []
        for key_cam in kernel_uv:
            key_cam = (key_cam[0], key_cam[1])
            kernel_xyz = kernel_cam2pc_dict[key_cam]
            kernel_pcd.append(kernel_xyz)
            # print(kernel_xyz)
        kernel_pcd = np.array(kernel_pcd)
        kernel_polar = convert_polar(kernel_pcd)
        # print(a)
        # print(kernel_pcd)
        # print(kernel_polar)

        return kernel_uv, kernel_pcd, kernel_polar

    def process_kernel_polar(self, kernel_polar):
        # print(kernel_polar)
        kernel_polar_T = kernel_polar.T

        theta, rho = kernel_polar_T
        # print(theta,rho)
        # rho_r = np.reshape(rho,(1,-1))
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(kernel_polar)
        # print(kmeans.cluster_centers_)
        # print(kmeans.labels_)
        # print('-------------------------------------------------------------------------')
        # print(kmeans.cluster_centers_[:,1])

        zero_key = kmeans.cluster_centers_[:, 1][0]
        one_key = kmeans.cluster_centers_[:, 1][1]

        # print(zero_key,one_key)

        # selecting low and high values
        # low = kmeans.cluster_centers_[:,1][0]

        select_lower = kmeans.labels_.astype(bool)
        select_higher = np.logical_not(select_lower)

        if zero_key < one_key:
            select_lower, select_higher = select_higher, select_lower

        kernel_low = kernel_polar[select_lower][:, 1]
        kernel_high = kernel_polar[select_higher][:, 1]

        # print(kernel_low,kernel_high)
        # print(kernel_low.max())
        # print(kernel_high.min())
        kernel_theta = kernel_polar[:, 0]
        res_rho_range = [kernel_low.max(), kernel_high.min()]
        res_theta_range = [kernel_theta.min(), kernel_theta.max()]

        # print(res_theta_range, res_rho_range)
        return res_theta_range, res_rho_range

    def kernel_plot_dict(self):
        """generating kernel size in polar as key, distance as value

        Returns:
            plot_dict: dictionary
        """
        plot_dict = {}
        # print(self.np_loc)

        # print("debuging line 311")
        for loc in self.np_loc:
            _, _, kernel_polar = self.post_kernel_method(loc)
            # print(kernel_polar.shape)
            if kernel_polar.shape[0] < 4:
                continue
            theta_range, rho_range = self.process_kernel_polar(kernel_polar)

            # for changing

            theta_key = (theta_range[0], theta_range[1])
            plot_dict[theta_key] = rho_range
        # print(plot_dict)
        return plot_dict


import argparse
import utils


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    # file_config
    parser.add_argument(
        "--file_name",
        type=str,
        required=False,
        default="aaa",
        help="filename for path to processing file",
    )
    parser.add_argument(
        "--test_folder",
        type=str,
        default="test_data",
        help="directory for occlusion area conversion",
    )
    parser.add_argument(
        "--bin_path",
        type=str,
        default="0000000000.bin",
        help="path for point cloud file",
    )
    parser.add_argument(
        "--img_path", type=str, default="0000000000.png", help="path for image file"
    )
    args = parser.parse_args()

    return args


args = parse_args_and_config()
from utils import Depth_converter, Kernel_tool

a = Depth_converter(args)
new_velo, new_cam = a.get_mapped_points()

kernel_tool = Kernel_tool(new_velo, new_cam)
key_u, key_v, out = kernel_tool.kernel_method(debug=True)
cam2pc_dict, pc2cam_dict = kernel_tool.get_mapping_dict()
kernel_cam2pc_dict = kernel_tool.get_kernel_dict(key_v, key_u)

plot_dict = kernel_tool.kernel_plot_dict()


polar_pcd = kernel_tool.polar_pcd
