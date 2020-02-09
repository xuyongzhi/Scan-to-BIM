#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py

#ipython tools/train.py -- ${CONFIG} 

CONFIG=configs/strpoints/strpoints_r50_fpn_1x_A.py
./tools/dist_train.sh ${CONFIG} 2

CONFIG=configs/strpoints/strpoints_r50_fpn_1x_B.py
#./tools/dist_train.sh ${CONFIG} 2

#CONFIG=configs/strpoints/strpoints_r50_fpn_1x_C.py
#./tools/dist_train.sh ${CONFIG} 2
#
#CONFIG=configs/strpoints/strpoints_r50_fpn_1x_D.py
#./tools/dist_train.sh ${CONFIG} 2
