#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py

#ipython tools/train.py -- ${CONFIG} 
#./tools/dist_train.sh ${CONFIG} 2

CONFIG=configs/strpoints/strpoints_r50_fpn_1x_lr5.py
#./tools/dist_train.sh ${CONFIG} 2

CONFIG=configs/strpoints/strpoints_r50_fpn_1x_lr05.py
#./tools/dist_train.sh ${CONFIG} 2

