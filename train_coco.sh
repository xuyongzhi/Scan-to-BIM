export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/reppoints/d_reppoints_moment_r50_fpn_2x.py
#CONFIG=configs/reppoints/reppoints_moment_r50_fpn_2x.py
#CONFIG=configs/reppoints/reppoints_moment_r101_dcn_fpn_2x.py



ipython tools/train.py -- ${CONFIG}  --cls refine_final
#./tools/dist_train.sh ${CONFIG} 2
