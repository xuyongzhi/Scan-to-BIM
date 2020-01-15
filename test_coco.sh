CONFIG=configs/reppoints/d_reppoints_moment_r50_fpn_2x.py
CHECKPOINT=checkpoints/reppoints_moment_r50_fpn_2x.pth


#python tools/test.py  $CONFIG  $CHECKPOINT --show
python tools/test.py  $CONFIG  $CHECKPOINT --out ./res/reppoints_moment_r50_fpn_2x.pickle  --eval bbox 

