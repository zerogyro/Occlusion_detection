from sensor_fusion_utils import get_mapped_points, get_sparse_depthmap, get_cam2velo_int
from keypoints_detector import get_keypoints_d
from utils import vis_keypoints_in_depth
import numpy as np
import cv2

bin_path = "../test_data/0000000000.bin"
img_path = "../test_data/0000000000.png"

new_velo, new_cam = get_mapped_points(bin_path, img_path)
s_dmap = get_sparse_depthmap(new_cam)
cam2velo_dict = get_cam2velo_int(new_velo, new_cam)

keypoints_d = get_keypoints_d(s_dmap)
keypoints_d = np.array(keypoints_d)


from utils import img_loader

path = "../test_data/0000000000.png"
img = img_loader(path)

from utils import vis_keypoints_on_rgb

img = vis_keypoints_on_rgb(img, keypoints_d)
cv2.imshow("test", img)
