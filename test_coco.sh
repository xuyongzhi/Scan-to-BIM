
CONFIG=configs/reppoints/d_reppoints_moment_r50_fpn_2x.py
CONFIG=configs/reppoints/d2_reppoints_moment_r50_fpn_2x.py

wdir=work_dirs/2_reppoints_moment_r50_fpn_2x_refine_bs2_lr10
CHECKPOINT=${wdir}/best.pth


ipython tools/test.py --  $CONFIG  $CHECKPOINT --show --cls refine
#python tools/test.py  $CONFIG  $CHECKPOINT --out ./res/reppoints_moment_r50_fpn_2x.pickle  --eval bbox 

