#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
CONFIG=configs/strpoints/strpoints_r50_fpn_1x_nor.py

ipython tools/train.py -- ${CONFIG} 

#./tools/dist_train.sh ${CONFIG} 2

CONFIG=configs/strpoints/strpoints_r50_fpn_1x_B.py
#./tools/dist_train.sh ${CONFIG} 2

CONFIG=configs/strpoints/strpoints_r50_fpn_1x_slr.py
#./tools/dist_train.sh ${CONFIG} 2
