

For your first query regarding x and y dimension there are two explanation.

Reason 1.

    For image re-projection pin hole camera model is used which is in perspective coordinate or homogenous coordinate. Perspective projection uses the image origin as centre of projection and points are mapped to the plane z=1. A 3D point [x y z] is represented by [xw yw zw w] and the point it maps on the plane is represented by [xw yw zw]. Normalising with w gives.

    So (x,y) -> [x y 1]T : Homogeneous Image Coordinates

    and (x,y,z) - > [x y z 1] T : Homogeneous Scene Coordinates

Reason 2.

    With respect to the paper you have attached, considering equation (4) and (5)

    enter image description here

    enter image description here

    It is clear that P is of dimension 3X4 and R is expanded to 4x4 dimension.Also x is of dimension 1x4. So as per matrix multiplication rule number of columns of first matrix must equal to the number of rows of second matrix. So for given P of 3x4 and R of 4x4, x has to be 1x4.

Now coming to your second question of LiDAR image fusion, It requires intrinsic and extrinsic parameters (relative rotation and translation) and camera matrix. This rotation and translation forms a 3x4 matrix called as transformation matrix. So the point fusion equations becomes

[x y 1]^T = Transformation Matrix * Camera Matrix * [X Y Z 1]^T




https://medium.com/swlh/camera-lidar-projection-navigating-between-2d-and-3d-911c78167a94


https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f


https://zhuanlan.zhihu.com/p/291644762

https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f

https://math.stackexchange.com/questions/1003801/inverse-of-an-invertible-upper-triangular-matrix-of-order-3
















https://zhuanlan.zhihu.com/p/78798251
clustering method
https://zhuanlan.zhihu.com/p/548058662

## notes for k=means
	1. Initialize k samples for center of clusters a = a1, a2, a3... ak
	2. Calculating xi for k center points and assign points in its cluster
	3. for each cluster aj, recalculate the cluster point aj = 1/ci sum x
	4. repeat 23






## Psudo code for kmeans

	Acquiring n data with m dimension
	Randomly generate k data with m dimension
	while(t):
		for i in n:
			for j in k:
				calculating i to j Euclidean distance		
		for i in k:
			1. find the cluster points
			2. change center points of data
	end














