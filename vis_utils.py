import numpy as np
import matplotlib.pyplot as plt
import cv2


def polar_plot(pcd_polar, plot_dict):
    HEIGHT = 576  # 48x12 (rho)
    WIDTH = 672  # 84 x 8
    LEFT_BOARDER = 48
    OUT_PATH = "eee.jpg"
    blank_image = np.ones((HEIGHT, WIDTH), np.uint8)
    blank_image = blank_image * 255

    def plot_polar_point_raw(polar_occupancy_point, canvas):
        for i, p in enumerate(polar_occupancy_point):
            theta = p[0]
            rho = p[1]
            # print(theta, rho)
            u = HEIGHT - int(rho * 12)
            v = int((theta - LEFT_BOARDER) * 8)
            canvas[u][v] = np.array(128)
        return canvas

    canvas = plot_polar_point_raw(pcd_polar, blank_image)

    def plot_2(canvas):
        for i, (k, v) in enumerate(plot_dict.items()):
            theta_range = k
            rho_range = v

            u_low = HEIGHT - int(rho_range[0] * 12)
            u_high = HEIGHT - int(rho_range[1] * 12)
            v_left = int((theta_range[0] - LEFT_BOARDER) * 8)
            v_right = int((theta_range[1] - LEFT_BOARDER) * 8)

            # theta = p[0]
            # rho = p[1]
            # #print(theta, rho)
            # u = HEIGHT- int(rho*12)
            # v = int((theta-LEFT_BOARDER)*8)
            check = canvas[u_high:u_low, v_left:v_right]
            check_flag = np.count_nonzero(check == 128)

            # generate selection

            # TODO: on BEV
            if check_flag < 10:
                canvas[u_high:u_low, v_left:v_right] = np.array(0)
        return canvas

    res_canvas = plot_2(canvas)
    cv2.imwrite(OUT_PATH, res_canvas)


from utils import polar_pcd, plot_dict


polar_plot(polar_pcd, plot_dict)
