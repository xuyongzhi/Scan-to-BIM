import os, glob, random, shutil
import numpy as np
IMAGE_SIZE =  512
TOPVIEW = 'TopView_All'
TOPVIEW = 'TopView_VerD'



IMG_PATH = f'/DT/BEIKE_Floorplan/processed_{IMAGE_SIZE}/{TOPVIEW}'
TRAIN_FILE_PATH = f'/DT/BEIKE_Floorplan/processed_{IMAGE_SIZE}/{TOPVIEW}/train.txt'
TEST_FILE_PATH = f'/DT/BEIKE_Floorplan/processed_{IMAGE_SIZE}/{TOPVIEW}/test.txt'
def gen_split_scene_list():
  files = glob.glob(IMG_PATH+'/*.npy')
  scenes = [os.path.basename(f.replace('.npy','')) for f in files]
  n = len(scenes)
  train_scenes = random.sample(scenes, 90)
  test_scenes = [s for s in scenes if s not in train_scenes]
  np.savetxt(TRAIN_FILE_PATH, train_scenes, '%s')
  np.savetxt(TEST_FILE_PATH, test_scenes, '%s')
  pass



ORG_PATH = f'/DT/BEIKE_Floorplan/processed_{IMAGE_SIZE}/{TOPVIEW}/test'
TRAIN_PATH = f'/DT/BEIKE_Floorplan/processed_{IMAGE_SIZE}/{TOPVIEW}/_train_90'
TEST_PATH = f'/DT/BEIKE_Floorplan/processed_{IMAGE_SIZE}/{TOPVIEW}/_test_10'

def split_and_mv(flag):
  train_path = TRAIN_PATH + '_' + flag
  test_path = TEST_PATH + '_' + flag
  if not os.path.exists(train_path):
    os.makedirs(train_path)
  if not os.path.exists(test_path):
    os.makedirs(test_path)

  files = glob.glob(ORG_PATH+'/*.npy')
  scenes = [os.path.basename(f.replace('.npy','')) for f in files]
  n = len(scenes)
  train_scenes = random.sample(scenes, 90)
  test_scenes = [s for s in scenes if s not in train_scenes]


  for s in train_scenes:
    shutil.copyfile(  os.path.join(ORG_PATH,    s+'.npy'),\
                      os.path.join(train_path,  s+'.npy'))

  for s in test_scenes:
    shutil.copyfile(  os.path.join(ORG_PATH,    s+'.npy'),\
                      os.path.join(test_path,  s+'.npy'))
  print(f'split ok: {flag}')

if __name__ == '__main__':
  #for flag in ['A', 'B']:
  #  split_and_mv( flag )
  gen_split_scene_list()
