import numpy as np
import matplotlib.pyplot as plt
import cv2

def vis_sparse_dmap(s_dmap):
    where_0 = np.where(s_dmap == 0)
    s_dmap = s_dmap / np.max(s_dmap)
    s_dmap[where_0] = 255 

    return s_dmap
    # plt.imshow(s_dmap, cmap='binary')
    # return 

def polar_plot(pcd_polar, plot_dict, gray_scale):
    HEIGHT = 576  # 48x12 (rho)
    WIDTH = 672  # 84 x 8
    LEFT_BOARDER = 48
    OUT_PATH = "occlusion_area.jpg"
    blank_image = np.ones((HEIGHT, WIDTH), np.uint8)
    blank_image = blank_image * 255

    def plot_polar_point_raw(polar_occupancy_point, canvas):
        for i, p in enumerate(polar_occupancy_point):
            theta = p[0]
            rho = p[1]
            # print(theta, rho)
            u = HEIGHT - int(rho * 12)
            v = int((theta - LEFT_BOARDER) * 8)
            canvas[u][v] = np.array(gray_scale)
        return canvas

    canvas = plot_polar_point_raw(pcd_polar, blank_image)
    ##Debug
    cv2.imwrite("canvas.png", canvas)

    def plot_2(canvas, post_process):
        for i, (k, v) in enumerate(plot_dict.items()):
            theta_range = k
            rho_range = v

            u_low = HEIGHT - int(rho_range[0] * 12)
            u_high = HEIGHT - int(rho_range[1] * 12)
            v_left = int((theta_range[0] - LEFT_BOARDER) * 8)
            v_right = int((theta_range[1] - LEFT_BOARDER) * 8)

            # check if the points are in empty area

            # if !, discard the points and not showing them
            if post_process:
                check = canvas[u_high:u_low, v_left:v_right]
                check_flag = np.count_nonzero(check == 100)

                if check_flag < 10:
                    canvas[u_high:u_low, v_left:v_right] = np.array(0)
            else:
                canvas[u_high:u_low, v_left:v_right] = np.array(0)
        return canvas

    # post processing is the extra function for eliminatin the noises in output
    res_canvas = plot_2(canvas, post_process=True)
    cv2.imwrite(OUT_PATH, res_canvas)
    print("visualization for polar index and occlusion area")


from utils import polar_pcd, plot_dict


polar_plot(polar_pcd, plot_dict, gray_scale=128)
