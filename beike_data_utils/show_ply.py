import os, glob
import numpy as np
from plyfile import PlyData, PlyElement
from tools.visual_utils import _show_objs_ls_points_ls, _show_3d_points_objs_ls

IMAGE_SIZE = 512

def load_ply(pcl_file):
    with open(pcl_file, 'rb') as f:
      plydata = PlyData.read(f)
    points = np.array(plydata['vertex'].data.tolist()).astype(np.float32)
    points = points[:,:9]
    assert points.shape[1] == 9
    points = flip_pcl(points)
    return points

def flip_pcl(points):
  x_min = points[:,0].min()
  x_max = points[:,0].max()
  x_mid = (x_min + x_max)/2
  points[:,0] = - points[:,0] + x_mid * 2
  return points

def cut_roof(points):
  z_min = points[:,2].min()
  z_max = points[:,2].max()
  z_th = z_min  + (z_max - z_min) * 0.7
  mask = points[:,2] < z_th
  return points[mask]

def show_pcl_f(pcl_file):
  points = load_ply(pcl_file)
  points = cut_roof(points)
  _show_3d_points_objs_ls( [ points[:,:3] ] ,  [points[:,3:6]])

def main():
  CUR_DIR = os.path.dirname(os.path.realpath(__file__))
  ROOT_DIR = os.path.dirname(CUR_DIR)
  base_dir = os.path.join(ROOT_DIR, f'data/beike/processed_{IMAGE_SIZE}' )

  ply_path = os.path.join(base_dir, 'ply')

  scene_file = os.path.join(base_dir, 'key_samples.txt')
  scenes = np.loadtxt(scene_file, dtype = str).tolist()
  #scenes = ['7w6zvVsOBAQK4h4Bne7caQ']
  #scenes = ['HBfvzptqI-PlRzcg3lnX-o']
  #scenes = ['auto3d-fpUiiote8TwrE68Zbrwsxu']
  scenes = ['kH3W13h2tW_4uZAf5JKmS2']
  scenes = ['18H6WOCclkJY34-TVuOqX3']
  n = len(scenes)
  assert n>0
  for i, s in enumerate(scenes):
    #if i<2:
    #  continue
    print(f'\n{i}')
    print(f'{s}')
    ply_f = os.path.join( ply_path, s + '.ply' )
    show_pcl_f( ply_f )
  pass

if __name__ == '__main__':
  main()
