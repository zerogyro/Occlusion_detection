import numpy as np
import os
import argparse
import math
from sklearn.cluster import KMeans
import matplotlib.image as mpimg


from sensor_fusion import get_mapped_points, get_sparse_depthmap, get_cam2velo_int
bin_path = 'test_data/0000000000.bin'
img_path = 'test_data/0000000000.png'
    
new_velo, new_cam = get_mapped_points(bin_path, img_path)
s_dmap = get_sparse_depthmap(new_cam)
cam2velo_dict = get_cam2velo_int(new_velo, new_cam)


IMG_H = 375
IMG_W = 1242
def general_kernel_process(s_dmap):
    # sliding window through whole sparse map
    # window_h, window_w, std_thresh



    img_h, img_w = s_dmap.shape
    assert img_h == IMG_H and img_w == IMG_W
    
    window_h = 5 
    window_w = 5 
    std_threshold = 2 
     
     
     
    res_keypoint = {}
    for r_s in range(0,img_h +1 - window_h):
        for c_s in range(0, img_w +1 - window_w):
            # retriving points within window
            window = s_dmap[r_s:r_s+window_h,c_s:c_s+window_w]
            # get !0 depth 
            new_window = window[np.where(window!=0)]

            
            if new_window.any():
                # getting keypoints
                # std:
                #std = sqrt(mean(x)), where x = abs(a - a.mean())**2.
                std_index = np.std(new_window)
    
                
                if std_index> std_threshold:
                    #print(new_window)
                    res_keypoint[(r_s, c_s)] = std_index
    return res_keypoint
             



#kernel tool for sparse map
def kernel_a(s_dmap):
    img_h, img_w = s_dmap.shape
    assert img_h == IMG_H and img_w == IMG_W
    
    window_h = 5 
    window_w = 5 
    std_threshold = 2
     
     
     
    for r_s in range(0,img_h +1 - window_h):
        for c_s in range(0, img_w +1 - window_w):
            # retriving points within window
            window = s_dmap[r_s:r_s+window_h,c_s:c_s+window_w]
            # get !0 depth 
            new_window = window[np.where(window!=0)]

            
            if new_window.any():
                # getting keypoints
                # std:
                #std = sqrt(mean(x)), where x = abs(a - a.mean())**2.
                std_index = np.std(new_window)
    
                
                if std_index> std_threshold:
                    #print(new_window)
                    #res_keypoint[(r_s, c_s)] = std_index
                    filter(window, r_s, c_s)
    
def filter(window, u , v):
    # this is already kernel window with huge std

    non_zero_index= np.where(window!=0)
    print(u,v)
    print(window)
    print(non_zero_index)
    
    exit()
    
def vis_kernel(res_keypoint, s_dmap):
    new_sdmap = s_dmap
    for key in res_keypoint.keys():
        new_sdmap[key[0], key[1]] = 128
    return new_sdmap
    





 
#res_keypoint = kernel_process(s_dmap)
res_keypoint = kernel_a(s_dmap)