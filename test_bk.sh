export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
CHECKPOINT=checkpoints/strpoints_moment_r50_fpn_1x.pth


ipython tools/test.py --  $CONFIG  $CHECKPOINT --show
#ipython tools/test.py --  $CONFIG  $CHECKPOINT --out ./res/strpoints_moment_r50_fpn_1x.pickle --eval bbox 

