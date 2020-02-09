#export CUDA_VISIBLE_DEVICES=1
DATA=D
IMAGE_SIZE=1024
CONFIG=configs/strpoints/strpoints_r50_fpn_1x_${DATA}.py
CHECKPOINT=work_dirs/lscope_istopleft_strpoints_moment_r50_fpn_1x_${DATA}_${IMAGE_SIZE}/best.pth

#CONFIG=configs/strpoints/strpoints_r50_fpn_1x_sd.py


ipython tools/test.py --  $CONFIG  $CHECKPOINT --show
#ipython tools/test.py --  $CONFIG  $CHECKPOINT --out ./res/strpoints_moment_r50_fpn_1x.pickle --eval bbox 

