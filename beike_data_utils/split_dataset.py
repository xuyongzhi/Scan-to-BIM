import os, glob, random, shutil
IMAGE_SIZE = 1024

ORG_PATH = f'/DT/BEIKE_Floorplan/processed_{IMAGE_SIZE}/topview/test'
TRAIN_PATH = f'/DT/BEIKE_Floorplan/processed_{IMAGE_SIZE}/topview/_train_89'
TEST_PATH = f'/DT/BEIKE_Floorplan/processed_{IMAGE_SIZE}/topview/_test_10'


BAD_SCENES =  ['7w6zvVsOBAQK4h4Bne7caQ', 'IDZkUGse-74FIy2OqM2u_Y', 'B9Abt6B78a0j2eRcygHjqC']

def split(flag):
  train_path = TRAIN_PATH + '_' + flag
  test_path = TEST_PATH + '_' + flag
  if not os.path.exists(train_path):
    os.makedirs(train_path)
  if not os.path.exists(test_path):
    os.makedirs(test_path)

  files = glob.glob(ORG_PATH+'/*.npy')
  scenes = [os.path.basename(f.replace('.npy','')) for f in files]
  scenes = [s for s in scenes if s not in BAD_SCENES]
  n = len(scenes)
  train_scenes = random.sample(scenes, 89)
  test_scenes = [s for s in scenes if s not in train_scenes]


  for s in train_scenes:
    shutil.copyfile(  os.path.join(ORG_PATH,    s+'.npy'),\
                      os.path.join(train_path,  s+'.npy'))

  for s in test_scenes:
    shutil.copyfile(  os.path.join(ORG_PATH,    s+'.npy'),\
                      os.path.join(test_path,  s+'.npy'))
  print(f'split ok: {flag}')

if __name__ == '__main__':
  for flag in ['A', 'B', 'C']:
    split( flag )
