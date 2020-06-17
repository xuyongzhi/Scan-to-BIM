import os
from shutil import copyfile

scenes = [ 'auto3d-fDNvVCEaDs3oniGpsmXo8k',  'auto3d-fpUiiote8TwrE68Zbrwsxu',  'auto3d-Nf1tEhOFifoJyX8OndX-sy',  'auto3d-noxPqe302We2byPCOYMPV_',  'auto3d-qbgc1aG14FpQhnmRd_yg8X', ]


path = 'data/beike/processed_512'
dst_path = os.path.join(path, 'SmallSamples')
if not os.path.exists(dst_path):
  os.makedirs(dst_path)

folder_ls = ['TopView_VerD', 'room_bboxes', 'relations', 'json']
formats = ['.npy', '.txt', '.npy', '.json']


for scene in scenes:
  for fold, fmt in zip(folder_ls, formats):
    src = os.path.join(path, fold, scene+'.npy')
    dst_fold = os.path.join(dst_path, fold)
    if not os.path.exists(dst_fold):
        os.makedirs(dst_fold)
    dst = os.path.join(dst_fold, scene + fmt)
    copyfile(src, dst)
print('finish')

