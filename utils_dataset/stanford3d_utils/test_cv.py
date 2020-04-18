import numpy as np
import cv2


from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls


'''
The angle output of cv2.minAreaRect:
  1. angle with x
  2. from x_min to x_max
  3. clock wise is positive
'''

img_size = (512,512)

point_inds = np.array([ [20,20], [500,100]])
#point_inds = np.array([ [500,20], [500,100]])
#point_inds = np.array([ [20,500], [500,100]])

point_inds = np.array([ [20,20], [500,500]])

point_inds = np.array([ [20,20], [500,500], [100, 90]])

#point_inds = np.array([ [20,500], [500,20], [220, 200]])

box2d = cv2.minAreaRect(point_inds)
img = np.zeros(img_size, dtype=np.uint8)

box2d = np.array( box2d[0]+box2d[1]+(box2d[2],) )[None, :]
print(box2d)
box2d[:,-1] *= np.pi / 180

_show_objs_ls_points_ls(img, [box2d], obj_rep='RoBox2D_CenSizeAngle', points_ls= [point_inds])

