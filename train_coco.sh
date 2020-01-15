#CONFIG=configs/reppoints/d_reppoints_moment_r50_fpn_2x.py
CONFIG=configs/reppoints/reppoints_moment_r50_fpn_2x.py
CONFIG=configs/reppoints/reppoints_moment_r101_dcn_fpn_2x.py



#ipython tools/train.py -- ${CONFIG} 
./tools/dist_train.sh ${CONFIG} 2
