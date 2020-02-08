#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
DATA=D
CHECKPOINT=work_dirs/${DATA}_lscope_istopleft_strpoints_moment_r50_fpn_1x//best.pth

#CONFIG=configs/strpoints/strpoints_r50_fpn_1x_sd.py


ipython tools/test.py --  $CONFIG  $CHECKPOINT --show
#ipython tools/test.py --  $CONFIG  $CHECKPOINT --out ./res/strpoints_moment_r50_fpn_1x.pickle --eval bbox 

