export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py

CP=./work_dirs/T90_r50_fpn_lscope_istopleft_refine_512_VerD_bs3_lr10_RA_Normrawstd_ChmOnly/best.pth
CP=./checkpoints/refine_best_20Feb.pth

ipython tools/train.py --  ${CONFIG} --rotate 1 --lr 0.01 --cls refine --corhm 0 

#./tools/dist_train.sh ${CONFIG} 2  --rotate 1 --lr 0.01  --cls refine --corhm 0
