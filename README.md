# Occlusion_detection

## Task Description
General Discription. this project is to automatically detect the occlusion area from point cloud and camera.

1.      Depth conversion: convert point cloud to camera view: 
            By Linear conversion
            Visualize dimension of the point cloud
2.      Detect key points in each kernel:
            Definition of key points
            Sliding window:
                this function needs to return kernel, and points within the kernel
            Alpha filter: distribution within the kernel   
            Beta filter: for each point, partial derivatives are computed with respect
3.      Visualization of results:
            Birds eye view
            Polar conversion
4.      Neural network simulation
                
                

## ALpha Kernel Processing
filtering keypoints by distribution / covariance
Depth Image/
Conversion of points
## Beta Kernel Processing
filtering keypoints by partial derivatives




## Detailed Descriptin of Beta filter: 


When $p_{uv}$ is the depth value of the pixel
its gradient with respect to its corresponding neighbors should be small enough.



## file discription
kernel_utils.py --key_u,key_v, out: filtering the keypoints and return u,v and out



**task list TODO**
- [x] Depth conversion in tool 
- [ ] Detecting key points using filter A
- [ ] Detecting key points using filter B
- [ ] run with one argument 
- [ ] 



----------------------------------------------------------------
OD_comp : vis_3d: tools for using open3d to visualize point clouds