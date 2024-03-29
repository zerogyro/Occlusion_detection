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

# kernel_dict = kernel_tool.get_kernel_dict(key_u, key_v)
# print(kernel_dict)
# if __name__ == "__main__":
#     args = parse_args_and_config()
#     from utils import Depth_converter, Kernel_tool

#     a = Depth_converter(args)
#     new_velo, new_cam = a.get_mapped_points()

#     kernel_tool = Kernel_tool(new_velo, new_cam)
#     key_u, key_v, out = kernel_tool.kernel_method(debug=True)

#     print(len(key_u))
#     print(len(key_v))
#     print(type(key_u))
# print(key_u, key_v)
# print(a.P2.shape, a.R0_rect.shape, a.Tr_velo_to_cam.shape)
# Transition_matrix = a.P2 * a.R0_rect * a.Tr_velo_to_cam
# print(Transition_matrix)

# new_velo, new_cam = a.get_mapped_points()
# print(new_velo.shape, new_cam.shape)

# print(a.P2)
# depth_points = new_cam.T
# velo_points = new_velo.T
# d1 = depth_points[0]
# p1 = velo_points[0]
# print(d1, p1)
# # for a in depth_points:
# #     print(a)

# # depth_cam = new_cam.T
# # for a in depth_cam:
# #     print(a)

# fx = 718.856
# fy = 718.856
# cx = 607.1928
# cy = 185.2157

# u, v, d = d1
# x, y, z, i = p1

# print(u / fx)
# print(x)
