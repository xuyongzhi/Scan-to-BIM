import os, glob, random, shutil
import numpy as np

def gen_split_scene_list():
  IMAGE_SIZE =  512

  cur_path = cur_dir = os.path.dirname(os.path.realpath(__file__))
  root_dir = os.path.dirname(cur_dir)
  data_path = os.path.join(root_dir, f'data/beike/processed_{IMAGE_SIZE}' )
  ply_path = os.path.join(data_path, 'ply')

  train_split_path = os.path.join(data_path, 'train.txt')
  test_split_path = os.path.join(data_path, 'test.txt')
  all_split_path = os.path.join(data_path, 'all.txt')


  files = glob.glob(ply_path+'/*.ply')
  scenes = [os.path.basename(f.replace('.ply','')) for f in files]
  n = len(scenes)
  assert n>0, "no file found"
  train_scenes = random.sample(scenes, min(n,90))
  test_scenes = [s for s in scenes if s not in train_scenes]

  np.savetxt(test_split_path, test_scenes, '%s')
  np.savetxt(train_split_path, train_scenes, '%s')
  np.savetxt(all_split_path, train_scenes + test_scenes, '%s')
  print(f'split ok')
  pass


if __name__ == '__main__':
  #for flag in ['A', 'B']:
  #  split_and_mv( flag )
  gen_split_scene_list()
