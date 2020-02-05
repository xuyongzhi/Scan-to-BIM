#export CUDA_VISIBLE_DEVICES=1
#CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
CONFIG=configs/strpoints/strpoints_r50_fpn_1x_sd.py


#ipython tools/train.py -- ${CONFIG} 
./tools/dist_train.sh ${CONFIG} 2
