# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
CONFIG=configs/strpoints/strpoints_r50_fpn_1x_Fpn34.py

#CP=./checkpoints/refine_best_20Feb.pth
wkdir=T90_r50_fpn_lscope_istopleft_refine_final_512_VerD_bs6_lr10_RA_Normrawstd_ChmR2P1N1_Rfiou740
CP=./work_dirs/${wkdir}/best.pth

ROTATE=1
CLS=refine_final
CORHM=1
DCN_ZERO_BASE=0

ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE --resume $CP

#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE



STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'

#ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE $STYLE


