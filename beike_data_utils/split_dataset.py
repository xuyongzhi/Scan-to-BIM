import os, glob, random, shutil

ORG_PATH = '/DT/BEIKE_Floorplan/processed_1024/images/public100_1024'
TRAIN_PATH = '/DT/BEIKE_Floorplan/processed_1024/images/_train_87'
TEST_PATH = '/DT/BEIKE_Floorplan/processed_1024/images/_test_10'


BAD_SCENES =  ['7w6zvVsOBAQK4h4Bne7caQ', 'IDZkUGse-74FIy2OqM2u_Y', 'B9Abt6B78a0j2eRcygHjqC']

def split(flag):
  train_path = TRAIN_PATH + '_' + flag
  test_path = TEST_PATH + '_' + flag
  if not os.path.exists(train_path):
    os.makedirs(train_path)
  if not os.path.exists(test_path):
    os.makedirs(test_path)

  files = glob.glob(ORG_PATH+'/*.density.png')
  scenes = [os.path.basename(f.replace('.density.png','')) for f in files]
  scenes = [s for s in scenes if s not in BAD_SCENES]
  n = len(scenes)
  train_scenes = random.sample(scenes, 87)
  test_scenes = [s for s in scenes if s not in train_scenes]


  for s in train_scenes:
    shutil.copyfile(  os.path.join(ORG_PATH,    s+'.density.png'),\
                      os.path.join(train_path,  s+'.density.png'))
    shutil.copyfile(  os.path.join(ORG_PATH,    s+'.norm.png'),\
                      os.path.join(train_path,  s+'.norm.png'))

  for s in test_scenes:
    shutil.copyfile(  os.path.join(ORG_PATH,    s+'.density.png'),\
                      os.path.join(test_path,  s+'.density.png'))
    shutil.copyfile(  os.path.join(ORG_PATH,    s+'.norm.png'),\
                      os.path.join(test_path,  s+'.norm.png'))
  print(f'split ok: {flag}')

if __name__ == '__main__':
  for flag in ['A', 'B', 'C', 'D', 'E']:
    split( flag )
