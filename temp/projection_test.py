from projection import P2, R0_rect, Tr_velo_to_cam
import numpy as np






m = P2*R0_rect*Tr_velo_to_cam

print(P2)
print(P2.shape)

# K = np.array([[0.58, 0, 0.5, 0],
# [0, 1.92, 0.5, 0],
# [0, 0, 1, 0],
# [0, 0, 0, 1]], dtype=np.float32)




fx = P2[0,0]
fy = P2[1,1]
cx = P2[0,2]
cy = P2[1,2]

print(fx,fy,cx,cy)


fx, fy, cx, cy = 0.58,1.92,0.5,0.5

a,b,c = 50.572 , 7.22 ,  1.937




u,v,d =501.31542795 ,151.90117302 ,50.33591657 

z = d
x1 = ((u-cx) * z) / fx
x2 = ((v-cy) * z) / fy
print(x1, x2, a)




xyz = np.array([50.572 , 7.22 ,  1.937  ,1.   ])
uvd = m* xyz.reshape((4,1))
print(uvd)
    #    z = depth_image[i][j]
    #    x = (j - CX_DEPTH) * z / FX_DEPTH
    #    y = (i - CY_DEPTH) * z / FY_DEPTH
    #    pcd.append([x, y, z])












# P2P= np.array([[ 7.183351e+02  ,0.000000e+00 , 6.003891e+02 , 0],
#  [ 0.000000e+00  ,7.183351e+02,  1.815122e+02, 0],
#  [ 0.000000e+00,  0.000000e+00 , 1.000000e+00 , 0]])


# print(P2P)


# mm = P2P * R0_rect*Tr_velo_to_cam


# bin_1 = np.array([50.572 , 7.22 ,  1.937  ,1.   ])
# bin_1 = bin_1.reshape((4,1))
# img_1 =np.array([501.31542795 ,151.90117302  ,50.33591657])


# check1 = m*bin_1
# check2 = mm*bin_1

# print(check1)
# print(check2)


