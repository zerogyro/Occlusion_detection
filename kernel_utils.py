import numpy as np
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt




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


def parse_args_and_config():

    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # file_config 
    parser.add_argument('--file_name', type=str, required=False, default = 'aaa',help='filename for path to processing file')

    args = parser.parse_args()

    return args



# initialize a kernel method class

class Kernel_tool(object):
    def __init__(self,args):
        #TODO: get_trainsition_matrix from args

        self.P2, self.R0_rect, self.Tr_velo_to_cam = get_transition_matrix()
    
    #TODO: get_mapped_points from args
    def get_mapped_points(self,binary_path, img_path):
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
        cam = self.P2 *  self.R0_rect * self.Tr_velo_to_cam * velo


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

    
    def debug(self):
        print(self.P2, self.R0_rect, self.Tr_velo_to_cam)
        

if __name__ == "__main__":

    
    k_tool = Kernel_tool(args= None)
    k_tool.debug()