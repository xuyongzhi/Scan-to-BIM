# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/_strpoints_r50_fpn_1x.py
CONFIG=configs/strpoints/_pcl_strpoints_r50_fpn_1x.py


wkdir=R50_fpn_refine_final_beike_pcl_2d_bs1_lr10_Daug_ChmR2P1N1_Rfiou743_Fpn34_Pbs1_Bp32_Vsz4Stem2
CP=./work_dirs/${wkdir}/best.pth

ROTATE=0
CLS=refine_final
CORHM=0
DCN_ZERO_BASE=0
BASE_PLANE=32
BS=4

ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE  --lr 0.03 --base_plane $BASE_PLANE   --bs 2
#--resume $CP 

#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE --lr 0.01 --base_plane $BASE_PLANE


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show

#ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE $STYLE --base_plane $BASE_PLANE


