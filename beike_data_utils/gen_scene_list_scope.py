import os, glob
import numpy as np
from plyfile import PlyData, PlyElement
from multiprocessing import Pool
from functools import partial
import random

IMAGE_SIZE = 512

def creat_ply_link(base_dir):
  base_dir = os.path.realpath(base_dir)
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
    print(f'creat dir: {base_dir}')
  root_dir = os.path.dirname(base_dir)
  src = os.path.join(root_dir, 'data', 'ply')
  dst = os.path.join(base_dir,  'ply')
  if not os.path.exists(dst):
    os.symlink(src, dst)
    print(f'make link: ply')
  src = os.path.join(root_dir, 'data', 'json')
  dst = os.path.join(base_dir,  'json')
  if not os.path.exists(dst):
    os.symlink(src, dst)
    print(f'make link: json')
  print('creat_ply_link ok')
  pass

def gen_scene_list(base_dir, scene_start=0, max_scene_num=None):
  ply_path = os.path.join(base_dir, 'ply')
  files = glob.glob(ply_path + '/*.ply')
  files = [f for f in files if os.path.exists( f.replace('ply', 'json') )]
  assert len(files) > 0
  scenes = [ os.path.basename(f).split('.')[0] for f in files ]
  scenes.sort()
  n0  = len(scenes)
  scene_start = max(0, scene_start)
  if max_scene_num is None:
    scene_end = n0
  else:
    scene_end = min(n0, scene_start + max_scene_num)
  scenes = scenes[scene_start : scene_end]
  n  = len(scenes)
  assert n>0

  scene_file = os.path.join(base_dir, 'all.txt')
  np.savetxt(scene_file, scenes, fmt='%s')

  print(f'Totally {n0} scenes found.')
  print(f'Save the names of \n\t\t{scene_start} - {scene_end}\n\t  in: {scene_file}')
  print(f'All the following steps will only process scenes listed in this file.')
  print(f'The scenes are sorted. So that repeat process does not matter.')
  print(f'\n\n')
  pass

def gen_train_test_split(base_dir):
  scene_file = os.path.join(base_dir, 'all.txt')
  scenes = np.loadtxt(scene_file, dtype = str).tolist()

  n  = len(scenes)
  assert n>0

  n_train = max(1, int(n*0.9))
  train_scenes = random.sample(scenes, n_train)
  test_scenes = [s for s in scenes if s not in train_scenes]

  train_scene_file = os.path.join(base_dir, 'train.txt')
  np.savetxt(train_scene_file, train_scenes, fmt='%s')

  test_scene_file = os.path.join(base_dir, 'test.txt')
  np.savetxt(test_scene_file, test_scenes, fmt='%s')
  print(f'\n\n')
  print(f'Update: {train_scene_file}')
  print(f'Update: {test_scene_file}')
  print(f'\n\n')
  pass

def get_scene_pcl_scopes(base_dir, pool_num=0):
  pcl_scope_dir = os.path.join(base_dir, 'pcl_scopes')
  if not os.path.exists(pcl_scope_dir):
    os.makedirs(pcl_scope_dir)
  scene_file = os.path.join(base_dir, 'all.txt')
  scenes = np.loadtxt(scene_file, dtype = str).tolist()
  n = len(scenes)
  assert n>0
  ply_path = os.path.join(base_dir, 'ply')
  if pool_num == 0:
    for i, scene in enumerate( scenes ):
      get_1_scene_pcl_scope(ply_path, pcl_scope_dir, scene)
      if i%2 == 0 or i==n-1:
        print(f'gen pcl scopes: {i}/{n}')
  else:
    func = partial( get_1_scene_pcl_scope, ply_path, pcl_scope_dir)
    with Pool(pool_num) as pool:
      pool.map(func, scenes)

  print(f'finished gen pcl scopes: {n}\n\n')

def get_1_scene_pcl_scope(ply_path, pcl_scope_dir, scene):
    scope_file = os.path.join(pcl_scope_dir, scene+'.txt')
    if os.path.exists(scope_file):
      return
    pcl_file = os.path.join(ply_path, scene+'.ply')

    points_data = load_ply(pcl_file)
    xyz_min = points_data[:,0:3].min(0, keepdims=True)
    xyz_max = points_data[:,0:3].max(0, keepdims=True)
    xyz_min_max = np.concatenate([xyz_min, xyz_max], 0)
    np.savetxt(scope_file, xyz_min_max, fmt='%.5f')
    print(f'Cal pcl scope: {scene}')

def load_ply(pcl_file):
    with open(pcl_file, 'rb') as f:
      plydata = PlyData.read(f)
    points = np.array(plydata['vertex'].data.tolist()).astype(np.float32)
    points = points[:,:9]
    assert points.shape[1] == 9
    return points

def gen_scene_list_pcl_scope(scene_start=0, max_scene_num = None, pool_num=0):
  CUR_DIR = os.path.dirname(os.path.realpath(__file__))
  ROOT_DIR = os.path.dirname(CUR_DIR)
  BASE_DIR = os.path.join(ROOT_DIR, f'data/beike/processed_{IMAGE_SIZE}' )

  creat_ply_link(BASE_DIR)
  gen_scene_list(BASE_DIR, scene_start, max_scene_num)
  gen_train_test_split(BASE_DIR)
  get_scene_pcl_scopes(BASE_DIR, pool_num)


if __name__ == '__main__':
  gen_scene_list_pcl_scope(scene_start=10, max_scene_num = 30, pool_num=3)


