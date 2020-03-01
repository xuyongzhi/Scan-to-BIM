#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py

CP=./work_dirs/T90_r50_fpn_lscope_istopleft_refine_final_512_VerD_bs3_lr10_RA_Normrawstd_Chm/best.pth
CP=./checkpoints/refine_best_20Feb.pth

ROTATE=1
CLS=refine_final
CORHM=1
DCN_ZERO_BASE=0
ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE
#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE

