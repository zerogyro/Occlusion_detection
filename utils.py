import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math




def get_transition_matrix(calib_file=None):
    '''Get transition
    input: calib_file
    output: transition matrix mapping from point cloud to image domain
    '''
    calib_file = 'calib/calib.txt'
    with open(calib_file, 'r') as f:
        calib = f.readlines()


    # P2 (3 x 4) for left eye
    P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
    
    
    return P2, R0_rect, Tr_velo_to_cam




P2, R0_rect, Tr_velo_to_cam = get_transition_matrix()


print(P2,R0_rect,Tr_velo_to_cam)
def get_mapped_points(binary_path, img_path):
    '''
    this function is to pre-processing the data from mapping point cloud points to image domain
    input: point cloud data path, img data path
    output: mapped points and corresponding image pixels
    '''
    png = mpimg.imread(img_path)
    IMG_H,IMG_W,_ = png.shape
    # read raw data from binary
    scan = np.fromfile(binary_path, dtype=np.float32).reshape((-1,4))
    points = scan[:, 0:3] # lidar xyz (front, left, up)
    #print(points[0])
    # TODO: use fov filter? 
    velo = np.insert(points,3,1,axis=1).T
    velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    cam = P2 * R0_rect * Tr_velo_to_cam * velo


    # 1 to 1 mapping for new_velo and cam
    cam_index = np.where(cam[2,:]>=0)[1]
    new_velo = np.take(velo, cam_index, axis=1)

    cam = np.delete(cam,np.where(cam[2,:]<0)[1],axis=1)
    # get u,v,z
    cam[:2] /= cam[2,:]



    u,v,z = cam
    u_out = np.logical_or(u<5, u>IMG_W-5)
    v_out = np.logical_or(v<5, v>IMG_H-5)
    outlier = np.logical_or(u_out, v_out)
    #print(outlier)

    

    new_velo = np.delete(new_velo,np.where(outlier),axis=1)
    new_cam = np.delete(cam,np.where(outlier),axis=1)
    new_cam = np.asarray(new_cam)
    # print(new_velo.shape, new_cam.shape)
    # print(type(new_velo),type(new_cam))
    return new_velo,new_cam,png

def get_mapping_dict(new_velo, new_cam):
    
    '''
    this function generate two mapping dictionaries from pcd2img and img2pcd
    input: new_velo, new_cam from preprocesing mapped points from point cloud data and image frame 
    output: cam2pc_dict, pc2cam_dict
    '''


    u,v,d = new_cam
    x,y,z,k = new_velo

    l = len(u)

    cam2pc_dict = {}
    pc2cam_dict = {}
    for i in range(l):

        cam2pc_dict[u[i],v[i]] = [x[i],y[i],z[i]]
        pc2cam_dict[x[i],y[i],z[i]] = [u[i],v[i]]
    return cam2pc_dict, pc2cam_dict



def sorting_frame(new_velo, new_cam):
    '''
    sort new_velo and new_cam according to the pixel location of the frame
    input: new_velo, new_cam after mapping
    output: new_velo, new_cam after sorting
    '''
    # print(new_cam.shape)
    # print(new_velo.shape)
    new_cam_t = new_cam.T
    new_velo_t = new_velo.T
    
    
    print(new_cam_t.shape)
    new_cam_t = np.array(new_cam_t)
    
    #a = np.array([(3, 2), (6, 2), (3, 6), (3, 4), (5, 3)])



    ind = np.lexsort((new_cam_t[:,1],new_cam_t[:,0]))    
    
    #print(new_cam_t[ind])
    #print(ind)
    
    
    
    res_new_cam_T = new_cam_t[ind]
    res_new_velo_T = new_velo_t[ind]

    
    
    ################################################################
    new_cam,new_velo = res_new_cam_T.T, res_new_velo_T.T
    return new_velo, new_cam

def get_range_from_loc(cam,loc,size,std_thresh,out,key_u,key_v):
    #out = []
    u,v,d = cam
    #x,y,z,k = c
    #print(b.shape, c.shape)
    mid_u, mid_v = loc

    u_in = np.logical_and(u>mid_u-size, u<mid_u+size)
    v_in = np.logical_and(v>mid_v-size, v<mid_v+size)
    inlier = np.logical_and(u_in, v_in)
    
    
    v = np.std(d[inlier])
    if v>std_thresh:
        print(loc)
        print("std:", v)
        key_u.append(loc[0])
        key_v.append(loc[1])
        out.append(v)
    #out = np.array(out)
    #print(out.shape)
    #print(inlier.shape)
    #return inlier

def filtering():
    
    out = []
    key_u = []
    key_v = []
    
    
    for i in new_cam_T:
        loc = i[:2]
        if gate(loc):
        #print(loc)
        # cam2pc_key = (loc[0],loc[1])
        # pc_v = cam2pc_dict[cam2pc_key]


            get_range_from_loc(new_cam,loc,4,3,out,key_u,key_v)
        #break


    return key_u, key_v, out






def get_range_from_loc_vis_distribution(cam,loc,size,std_thresh,out,key_u,key_v):
    #out = []
    u,v,d = cam
    #x,y,z,k = c
    #print(b.shape, c.shape)
    mid_u, mid_v = loc

    u_in = np.logical_and(u>mid_u-size, u<mid_u+size)
    v_in = np.logical_and(v>mid_v-size, v<mid_v+size)
    inlier = np.logical_and(u_in, v_in)
    
    
    v = np.std(d[inlier])
    if v<1 and len(d[inlier])>10:
        # print(loc)
        # print("std:", v)
        # key_u.append(loc[0])
        # key_v.append(loc[1])
        # out.append(v)
        print('++++++++++++++++++++++++++++++++++++++++++++++')
        print(d[inlier])
        # print(u[inlier])
        # print(v[inlier])
        
        print('++++++++++++++++++++++++++++++++++++++++++++++')
        exit()
    #out = np.array(out)
    #print(out.shape)
    #print(inlier.shape)
    #return inlier


def vis():
    out = []
    key_u = []
    key_v = []
    
    
    for i in new_cam_T:
        loc = i[:2]
        if gate(loc):
        #print(loc)
        # cam2pc_key = (loc[0],loc[1])
        # pc_v = cam2pc_dict[cam2pc_key]


            get_range_from_loc_vis_distribution(new_cam,loc,4,3,out,key_u,key_v)
        #break


    return key_u, key_v, out
def gate(loc, gate_size = -1.6):
    cam2pc_key = (loc[0],loc[1])
    pc_v = cam2pc_dict[cam2pc_key]
    pc_z = pc_v[2]
    #print(pc_z,'----------------------------------')
    if pc_z <-1.2:
        return False
    return True




#####################################################################################

binary_path, img_path = 'test_data/0000000000.bin', 'test_data/0000000000.png'
png = mpimg.imread(img_path)
IMG_H,IMG_W,_ = png.shape
clean = np.zeros((IMG_H,IMG_W,3))
new_velo, new_cam,png = get_mapped_points(binary_path, img_path)
new_velo, new_cam = sorting_frame(new_velo, new_cam)



new_velo_T, new_cam_T = new_velo.T, new_cam.T

cam2pc_dict, pc2cam_dict = get_mapping_dict(new_velo, new_cam)
    #print(cam2pc_dict.keys())
    
    
key_u,key_v,out = filtering()


#vis()





# print(key_u,key_v,out)
# print(len(key_u),len(key_v))