import numpy as np
from utils import key_v, key_u, cam2pc_dict
import math
import cv2
from sklearn.cluster import KMeans  






# retrive keypoints from processed point cloud points from utils

np_key_v = np.array(key_v)
np_key_u = np.array(key_u)
key_len = len(np_key_v)
assert key_len == len(np_key_u) 
np_loc = np.stack((np_key_u, np_key_v),axis=-1)




def get_kernel_dict():
    kernel_cam2pc_dict = {}
    for k in np_loc:
        #print(k)
        key = (k[0],k[1])
        v = cam2pc_dict[key]
        kernel_cam2pc_dict[key] = v
    #print(kernel_cam2pc_dict)
    return kernel_cam2pc_dict

kernel_cam2pc_dict = get_kernel_dict()

def convert_polar(np_pcd):
    np_xy = np.array(np_pcd[:,:-1],dtype=object)
    polar_index = []
    #print('start')
    check = []
    for pair in np_xy:
        r = math.sqrt(pair[0] ** 2 + pair[1] ** 2)
        theta = (math.atan2(pair[0], pair[1]) / math.pi) * 180
        polar_index.append([theta, r])
        #check.append([theta, r, pair[0], pair[1]])
    # print(polar_index)
    polar_index = np.array(polar_index)
    #check = np.array(check)
    return polar_index



# Reimplementation of utils sorting frame and filtering
# kernel_method
# process_kernel_method: KMEANS

def kernel_method(loc,size=4):
    #print('start kernel method')
    # first get kernel size
    # key_v 
    # key_u
    #print(len(key_u),len(key_v))
    size = 5
    
    
    #print('preprocessing kernel method for kernel_uv, kernel_pcd, kernel_polar')
    mid_u, mid_v = loc
    #print(mid_u, mid_v)
    
    #print(key_u[0],key_v[0])
    u_in = np.logical_and(key_u>mid_u-size, key_u<mid_u+size)
    v_in = np.logical_and(key_v>mid_v-size, key_v<mid_v+size)
    inlier = np.logical_and(u_in, v_in)
    
    
    #print(inlier)
    
    kernel_uv = np_loc[inlier]
    kernel_pcd = []
    for key_cam in kernel_uv:
        key_cam = (key_cam[0],key_cam[1])
        kernel_xyz = kernel_cam2pc_dict[key_cam]
        kernel_pcd.append(kernel_xyz)
        #print(kernel_xyz)
    kernel_pcd = np.array(kernel_pcd)
    kernel_polar = convert_polar(kernel_pcd)
    # print(a)
    # print(kernel_pcd)
    # print(kernel_polar)
    
    return kernel_uv,kernel_pcd, kernel_polar





def process_kernel_polar(kernel_polar):
    #print(kernel_polar)
    kernel_polar_T = kernel_polar.T
    
    theta,rho = kernel_polar_T
    #print(theta,rho)
    #rho_r = np.reshape(rho,(1,-1))
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(kernel_polar)
    # print(kmeans.cluster_centers_)
    # print(kmeans.labels_)
    # print('-------------------------------------------------------------------------')
    # print(kmeans.cluster_centers_[:,1])
    
    zero_key = kmeans.cluster_centers_[:,1][0]
    one_key = kmeans.cluster_centers_[:,1][1]
    
    #print(zero_key,one_key)
    
    # selecting low and high values
    #low = kmeans.cluster_centers_[:,1][0] 
    
    
    
    select_lower = kmeans.labels_.astype(bool)
    select_higher = np.logical_not(select_lower)
    
    if zero_key< one_key:
        select_lower,select_higher = select_higher,select_lower
    
    
    kernel_low = kernel_polar[select_lower][:,1]
    kernel_high = kernel_polar[select_higher][:,1]    
    
    
    
    # print(kernel_low,kernel_high)
    # print(kernel_low.max())
    # print(kernel_high.min())
    kernel_theta = kernel_polar[:,0]
    res_rho_range = [kernel_low.max(),kernel_high.min()]
    res_theta_range = [kernel_theta.min(),kernel_theta.max()]
    
    
    
    #print(res_theta_range, res_rho_range) 
    return res_theta_range, res_rho_range
    #print(kernel_low[:,1].)

    
    
    
def kernel_tool():
    plot_dict= {}
    for loc in np_loc:
        _,_,kernel_polar = kernel_method(loc)
        #print(kernel_polar.shape)
        if kernel_polar.shape[0]<4:
            continue
        theta_range,rho_range = process_kernel_polar(kernel_polar)
        
        
        # for changing
        
        theta_key = (theta_range[0],theta_range[1])
        plot_dict[theta_key] = rho_range
    #print(plot_dict)
    return plot_dict




plot_dict = kernel_tool()
if __name__ == '__main__':
    plot_dict = kernel_tool()
    print(type(plot_dict.keys()))
    print(len(plot_dict.values()), len(plot_dict.keys()))