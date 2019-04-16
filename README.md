# PCV_Assignment_07
Three-dimensional reconstruction
## 外极几何
  已知两个摄像头的光心OO和O′O′，PP为空间中的一点，pp和p′p′是点PP在两个摄像头成的像中的投影。  
  平面OO′POO′P称为外极平面，显然pp和p′p′是OPOP和OP′OP′上的，即该5点共面。外极平面OO′POO′P与两个相机的视平面相交于线ll和l′l′，这两条直线称为外极线。其中ll是与p′p′相关的外极线，l′l′是与pp相关的外极线。且pp在ll上，p′p′在l′l′上。OO和O′O′与相关视平面相交于点ee和e′e′，这两个点分别为OO和O′O′在对方视平面的投影。如下图所示：  
  ![emmmm]()
  
  按照上面的定义，假设pp和p′p′分别是空间中同一点在两个不同视平面上的像点，则p′p′一定在l′l′上，pp一定在ll上，这就是外极线约束。  
  如果已经知道相机的参数，那么在重建过程中遇到的问题就是两幅图像之间的关系，外极线约束的主要作用就是限制对应特征点的搜索范围，将对应特征点的搜索限制在极线上。
  
## 基础矩阵
  上面提到的是两幅图像之间的约束关系，这种约束关系使用代数的方式表示出来即为基本矩阵。假设M和M′分别为两个摄像机的参数矩阵，其他同上图所示。则极线l′的参数方程为：  
  
![emmmm]()
  
  
![emmmm]()
## 确定基础矩阵F



## 实施
  基本步骤：计算图像对的特征匹配，并估计基础矩阵。使用外极线作为第二个输入，通过在外极线上对每个特征点寻找最佳的匹配来找到更多的匹配。
  
代码：  
```python
# coding: utf-8
from PIL import Image
from numpy import *
from pylab import *
import numpy as np
from PCV.geometry import camera
from PCV.geometry import homography
from PCV.geometry import sfm
from PCV.localdescriptors import sift

# camera = reload(camera)
# homography = reload(homography)
# sfm = reload(sfm)
# sift = reload(sift)

# Read features
im1 = array(Image.open('./data/00A1.jpg'))
sift.process_image('./data/00A1.jpg', 'im1.sift')

im2 = array(Image.open('./data/00A2.jpg'))
sift.process_image('./data/00A2.jpg', 'im2.sift')

l1, d1 = sift.read_features_from_file('im1.sift')
l2, d2 = sift.read_features_from_file('im2.sift')

matches = sift.match_twosided(d1, d2)

ndx = matches.nonzero()[0]
x1 = homography.make_homog(l1[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2, :2].T)

d1n = d1[ndx]
d2n = d2[ndx2]
x1n = x1.copy()
x2n = x2.copy()

figure(figsize=(16,16))
sift.plot_matches(im1, im2, l1, l2, matches, True)
show()

#def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).

    input: x1, x2 (3*n arrays) points in hom. coordinates. """

    from PCV.tools import ransac
    data = np.vstack((x1, x2))
    d = 10 # 20 is the original
    # compute F and return with inlier index
    F, ransac_data = ransac.ransac(data.T, model,
                                   8, maxiter, match_threshold, d, return_all=True)
    return F, ransac_data['inliers']

# find F through RANSAC
model = sfm.RansacModel()
F, inliers = F_from_ransac(x1n, x2n, model, maxiter=5000, match_threshold=1e-3)
print(F)

P1 = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_fundamental(F)

print(P2)
print(F)

# P2, F (1e-4, d=20)
# [[ -1.48067422e+00   1.14802177e+01   5.62878044e+02   4.74418238e+03]
#  [  1.24802182e+01  -9.67640761e+01  -4.74418113e+03   5.62856097e+02]
#  [  2.16588305e-02   3.69220292e-03  -1.04831621e+02   1.00000000e+00]]
# [[ -1.14890281e-07   4.55171451e-06  -2.63063628e-03]
#  [ -1.26569570e-06   6.28095242e-07   2.03963649e-02]
#  [  1.25746499e-03  -2.19476910e-02   1.00000000e+00]]

# triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2)

# plot the projection of X
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2)
x1p = cam1.project(X)
x2p = cam2.project(X)

figure(figsize=(16, 16))
imj = sift.appendimages(im1, im2)
imj = vstack((imj, imj))

imshow(imj)

cols1 = im1.shape[1]
rows1 = im1.shape[0]
for i in range(len(x1p[0])):
    if (0<= x1p[0][i]<cols1) and (0<= x2p[0][i]<cols1) and (0<=x1p[1][i]<rows1) and (0<=x2p[1][i]<rows1):
        plot([x1p[0][i], x2p[0][i]+cols1],[x1p[1][i], x2p[1][i]],'c')
axis('off')
show()

d1p = d1n[inliers]
d2p = d2n[inliers]

# Read features
im3 = array(Image.open('./data/00A3.jpg'))
sift.process_image('./data/00A3.jpg', 'im3.sift')
l3, d3 = sift.read_features_from_file('im3.sift')


# In[36]:
matches13 = sift.match_twosided(d1p, d3)


# In[37]:
ndx_13 = matches13.nonzero()[0]
x1_13 = homography.make_homog(x1p[:, ndx_13])
ndx2_13 = [int(matches13[i]) for i in ndx_13]
x3_13 = homography.make_homog(l3[ndx2_13, :2].T)


# In[38]:
figure(figsize=(16, 16))
imj = sift.appendimages(im1, im3)
imj = vstack((imj, imj))

imshow(imj)

cols1 = im1.shape[1]
rows1 = im1.shape[0]
for i in range(len(x1_13[0])):
    if (0<= x1_13[0][i]<cols1) and (0<= x3_13[0][i]<cols1) and (0<=x1_13[1][i]<rows1) and (0<=x3_13[1][i]<rows1):
        plot([x1_13[0][i], x3_13[0][i]+cols1],[x1_13[1][i], x3_13[1][i]],'c')
axis('off')
show()


# In[39]:
P3 = sfm.compute_P(x3_13, X[:, ndx_13])


# In[40]:
print(P3)

# In[41]:
print(P1)
print(P2)
print(P3)


# In[22]:

# Can't tell the camera position because there's no calibration matrix (K)


```
### 室外
基础矩阵F:  
```
[[ 2.73120053e-08  7.74483398e-07 -8.61310304e-04]
 [-7.35236573e-07  4.79127204e-07  5.09280691e-03]
 [ 4.95072878e-04 -5.54368224e-03  1.00000000e+00]]
```
  
  
  
```
P2:
[[-7.90811598e-01  4.67595290e+00  9.18154113e+02  6.58990535e+03]
 [ 5.67595341e+00 -3.35611162e+01 -6.58990485e+03  9.18148569e+02]
 [ 5.07869581e-03  3.83245933e-03 -3.69868917e+01  1.00000000e+00]]
F:
 [[ 2.73120053e-08  7.74483398e-07 -8.61310304e-04]
 [-7.35236573e-07  4.79127204e-07  5.09280691e-03]
 [ 4.95072878e-04 -5.54368224e-03  1.00000000e+00]]
```
  

```
P3:
[[-7.85170378e-01  5.20471742e-01  4.26576118e-04 -4.78953315e-06]
 [-2.30244182e-01 -2.16751169e-01 -3.22336982e-04  1.19706715e-03]
 [-3.09656944e-04 -2.91510100e-04 -4.33513168e-07  1.60994364e-06]]
 
 P1:
[[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]]
 
 P2:
[[-7.90811598e-01  4.67595290e+00  9.18154113e+02  6.58990535e+03]
 [ 5.67595341e+00 -3.35611162e+01 -6.58990485e+03  9.18148569e+02]
 [ 5.07869581e-03  3.83245933e-03 -3.69868917e+01  1.00000000e+00]]
 
 P3:
[[-7.85170378e-01  5.20471742e-01  4.26576118e-04 -4.78953315e-06]
 [-2.30244182e-01 -2.16751169e-01 -3.22336982e-04  1.19706715e-03]
 [-3.09656944e-04 -2.91510100e-04 -4.33513168e-07  1.60994364e-06]]
 ```
 
 

### 室内
基础矩阵F:  
```
[[-4.51196382e-07  1.67478886e-05 -1.05720209e-02]
 [-1.71583442e-05 -6.82422759e-07  8.03187606e-03]
 [ 1.07119393e-02 -8.92502716e-03  1.00000000e+00]]
```
  
  
```
P2:
[[-6.44506578e+00  4.89649450e+00  6.09641567e+02  5.57745191e+02]
 [ 5.89649337e+00 -4.47975741e+00 -5.57734479e+02  6.09632642e+02]
 [ 9.61611839e-03  1.00796687e-02 -1.15082388e+01  1.00000000e+00]]
F:
[[-4.51196382e-07  1.67478886e-05 -1.05720209e-02]
 [-1.71583442e-05 -6.82422759e-07  8.03187606e-03]
 [ 1.07119393e-02 -8.92502716e-03  1.00000000e+00]]

```
  

```
P3:
[[-7.95693501e-03  7.42092498e-02 -1.57091634e-04  9.96781692e-01]
 [ 1.85933469e-03 -2.89388097e-02 -1.52622900e-04  1.09282148e-03]
 [-1.36126006e-05 -1.12621684e-04 -3.74576495e-07  2.07619233e-06]]
 
 P1:
[[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]]
 
 P2:
[[-6.44506578e+00  4.89649450e+00  6.09641567e+02  5.57745191e+02]
 [ 5.89649337e+00 -4.47975741e+00 -5.57734479e+02  6.09632642e+02]
 [ 9.61611839e-03  1.00796687e-02 -1.15082388e+01  1.00000000e+00]]
 
 P3:
[[-7.95693501e-03  7.42092498e-02 -1.57091634e-04  9.96781692e-01]
 [ 1.85933469e-03 -2.89388097e-02 -1.52622900e-04  1.09282148e-03]
 [-1.36126006e-05 -1.12621684e-04 -3.74576495e-07  2.07619233e-06]]

 ```
