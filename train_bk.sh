#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py

#ipython tools/train.py --  ${CONFIG} --rotate 0 --lr 0.01 --cls refine 

./tools/dist_train.sh ${CONFIG} 2  --rotate 1 --lr 0.01  --cls refine --dcn_zero_base 0 --corhm 0
