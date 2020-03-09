# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/_strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/strpoints_r50_fpn_1x_IoU844.py


wkdir=T90_r50_fpn_lscope_istopleft_refine_final_1024_VerD_bs1_lr50_RA_Normrawstd_ChmR2P1N1_Rfiou743_Fpn34_Pbs1
CP=./work_dirs/${wkdir}/best.pth

ROTATE=1
CLS=refine_final
CORHM=1
DCN_ZERO_BASE=0


#ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE 

./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE --lr 0.05


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show

#ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE $STYLE


