#export CUDA_VISIBLE_DEVICES=1
DATA=A
IMAGE_SIZE=512
CONFIG=configs/strpoints/strpoints_r50_fpn_1x_nor.py
workdir=lscope_istopleft_strpoints_moment_r50_fpn_1x_A_512_lr10_All_nr
CHECKPOINT=work_dirs/${workdir}/best.pth

#CONFIG=configs/strpoints/strpoints_r50_fpn_1x_sd.py


ipython tools/test.py --  $CONFIG  $CHECKPOINT --show
#ipython tools/test.py --  $CONFIG  $CHECKPOINT --out ./res/strpoints_moment_r50_fpn_1x.pickle --eval bbox 

