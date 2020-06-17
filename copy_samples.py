import os
import numpy as np
from shutil import copyfile

path = 'data/beike/processed_512'
scenes = [ 'auto3d-fDNvVCEaDs3oniGpsmXo8k',  'auto3d-fpUiiote8TwrE68Zbrwsxu',  'auto3d-Nf1tEhOFifoJyX8OndX-sy',  'auto3d-noxPqe302We2byPCOYMPV_',  'auto3d-qbgc1aG14FpQhnmRd_yg8X', ]
scenes = np.loadtxt( os.path.join(path, 'all.txt'), dtype=str ).tolist() [0:10]


dst_path = os.path.join(path, 'SmallSamples')
if not os.path.exists(dst_path):
  os.makedirs(dst_path)

folder_ls = ['TopView_VerD', 'room_bboxes', 'relations', 'json', 'pcl_scopes', 'ply']
formats = ['.npy', '.npy', '.npy', '.json', '.txt', '.ply']


for scene in scenes:
  for fold, fmt in zip(folder_ls, formats):
    src = os.path.join(path, fold, scene+fmt)
    dst_fold = os.path.join(dst_path, fold)
    if not os.path.exists(dst_fold):
        os.makedirs(dst_fold)
    dst = os.path.join(dst_fold, scene + fmt)
    copyfile(src, dst)
print(f'finish copy these scenes to\n\t {dst_path}')

