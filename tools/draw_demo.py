from tools.visual_utils import _draw_points, _draw_lines, draw_XYLgWsA
from obj_geo_utils.geometry_utils import four_corners_to_box_np, sort_four_corners_np
from obj_geo_utils.obj_utils import OBJ_REPS_PARSE
import cv2
import numpy as np
import mmcv

def draw_corners_to_box( point_thickness=25,
                         line_thickness=4,
                        box_thickness = 4):
  corners = np.array( [
    [100, 100],
    [800, 300],
    [900, 500],
    [600, 800],
  ] )
  corners = sort_four_corners_np(corners[None,:,:])[0].reshape(-1,2)
  center = corners.mean(0,keepdims=True)
  box_XYDAsinAsinSin2 = four_corners_to_box_np(corners[None,:,:])
  box = OBJ_REPS_PARSE.encode_obj(box_XYDAsinAsinSin2, 'XYDAsinAsinSin2', 'XYLgWsA')

  tmp = np.repeat( center, 4, 0)
  mid_points = ( corners + corners[[1,2,3,0],:] ) /2
  mid_lines = np.concatenate([mid_points[:,None,:], tmp[:,None,:]], 1)

  lines = np.concatenate([corners[:,None,:], tmp[:,None,:]], 1)


  h,w = 1024, 1024
  img = np.zeros([h,w,3], dtype=np.uint8) + 255
  img = _draw_points(img, corners, 'red', point_thickness)
  img = _draw_points(img, center, 'blue', point_thickness)
  img = _draw_points(img, mid_points, 'green', point_thickness)
  img = _draw_lines(img, lines, 'red', line_thickness)
  img = _draw_lines(img, mid_lines, 'blue', line_thickness)
  img = draw_XYLgWsA(img, box, 'black', box_thickness)


  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  cv2.imshow('', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  draw_corners_to_box()
