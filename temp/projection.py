import numpy as np
import matplotlib.image as mpimg
def get_transition_matrix():
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
m1 = P2 * R0_rect * Tr_velo_to_cam

K = np.array([[0.58, 0, 0.5, 0],
[0, 1.92, 0.5, 0],
[0, 0, 1, 0],
[0, 0, 0, 1]], dtype=np.float32)

Rt = R0_rect* Tr_velo_to_cam

m2 = K*Rt


print("#############P2")
print(P2)
print("#############R0")
print(R0_rect)

print("#############Tr_velocam")
print(Tr_velo_to_cam)


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



#https://github.com/nianticlabs/monodepth2/issues/43

# print(K)

# print(m1)



# pcd1 = np.array([6.26, 0.07, -1.695])
# img1 = np.array([604.1435187957974, 374.6665656671073])


# bin_1 = np.array([50.572 , 7.22 ,  1.937  ,1.   ])
# bin_1 = bin_1.reshape((4,1))
# img_1 =np.array([501.31542795 ,151.90117302  ,50.33591657])
# print(img_1)


# check = m1*bin_1
# check /= 50.3359






