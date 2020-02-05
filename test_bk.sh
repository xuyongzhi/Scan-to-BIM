#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
CHECKPOINT=work_dirs/strpoints_moment_r50_fpn_1x_debuging/best.pth
#CHECKPOINT=./checkpoints/strpoints_moment_r50_fpn_1x.pth

CONFIG=configs/strpoints/strpoints_r50_fpn_1x_sd.py
CHECKPOINT=work_dirs/2_line_scope_strpoints_moment_r50_fpn_1x_debuging/best.pth
CHECKPOINT=work_dirs/2_lscope_istopleft_strpoints_moment_r50_fpn_1x_debuging/best.pth


ipython tools/test.py --  $CONFIG  $CHECKPOINT --show
#ipython tools/test.py --  $CONFIG  $CHECKPOINT --out ./res/strpoints_moment_r50_fpn_1x.pickle --eval bbox 

