import os
import matplotlib.image as mpimg
import numpy as np

TEST_DATA_FOLDER = "test_data"


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


def loader(file_name):
    """
    return png and velo from file_name
    """
    bin_file_name = file_name + ".bin"
    img_file_name = file_name + ".png"
    binary_path = os.path.join(TEST_DATA_FOLDER, bin_file_name)
    img_path = os.path.join(TEST_DATA_FOLDER, img_file_name)

    png = mpimg.imread(img_path)
    scan = np.fromfile(binary_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    return png, velo


def get_mapped_points(png, velo):
    cam = P2 * R0_rect * Tr_velo_to_cam * velo
    IMG_W, IMG_H, _ = png.shape
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
