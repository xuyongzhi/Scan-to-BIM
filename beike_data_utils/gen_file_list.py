import os.path as osp
import glob
import numpy as np

path = '/home/z/Research/mmdetection/work_dirs/Backup/Good_8March/Good/Eval_Score4_corDis15_optGraph10_line_ave_10Imgs'
path = '/home/z/Research/mmdetection/work_dirs/Backup/Good_8March/Good/Eval_Score4_corDis15_optGraph10_composite_90Imgs'
files = glob.glob(path + '/*EvalGt.png')
files.sort()
files = [f.split('_wall')[0] for f in files]
files = [f.split('/')[-1] for f in files]
files = np.unique(files)

filename = osp.join(path + '/file.txt')
np.savetxt(filename, files, fmt='%s')

pass
