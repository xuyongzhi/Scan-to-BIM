# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/pcl_strpoints_r50_fpn_1x.py
CONFIG=configs/strpoints/pcl_sparse3d_strpoints_r50_fpn_1x.py


wkdir=TPV_r50_fpn_refine_final_beike2d_bs1_lr10_512_VerD_RA_Normrawstd_Rfiou743_Fpn35_Pbs1_Bp32
CP=./work_dirs/${wkdir}/best.pth
#CONFIG=./work_dirs/${wkdir}/_pcl_strpoints_r50_fpn_1x.py


ROTATE=0
CLS=refine_final
CORHM=0
DCN_ZERO_BASE=0
BASE_PLANE=32
BS=1

ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE  --lr 0.05 --base_plane $BASE_PLANE --bs $BS 
#--resume $CP 

#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE --lr 0.01 --base_plane $BASE_PLANE --bs $BS


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show

#ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE $STYLE --base_plane $BASE_PLANE


