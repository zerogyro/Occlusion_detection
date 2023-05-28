from projection import P2, R0_rect, Tr_velo_to_cam
import numpy as np






m = P2*R0_rect*Tr_velo_to_cam



P2P= np.array([[ 7.183351e+02  ,0.000000e+00 , 6.003891e+02 , 0],
 [ 0.000000e+00  ,7.183351e+02,  1.815122e+02, 0],
 [ 0.000000e+00,  0.000000e+00 , 1.000000e+00 , 0]])


print(P2P)


mm = P2P * R0_rect*Tr_velo_to_cam


bin_1 = np.array([50.572 , 7.22 ,  1.937  ,1.   ])
bin_1 = bin_1.reshape((4,1))
img_1 =np.array([501.31542795 ,151.90117302  ,50.33591657])


check1 = m*bin_1
check2 = mm*bin_1

print(check1)
print(check2)


