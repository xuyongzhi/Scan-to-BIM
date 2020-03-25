# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/_strpoints_r50_fpn_1x.py
CONFIG=configs/strpoints/_pcl_strpoints_r50_fpn_1x.py


wkdir=T90_r50_fpn_lscope_istopleft_refine_final_512_VerD_beike_pcl_2d_bs4_lr10_ChmR2P1N1_Rfiou743_Fpn35_Pbs1_Bp32
#wkdir=T90_r50_fpn_lscope_istopleft_refine_final_512_VerD_beike2d_bs4_lr10_RA_Normrawstd_ChmR2P1N1_Rfiou743_Fpn35_Pbs1_Bp64
CP=./work_dirs/${wkdir}/best.pth

ROTATE=1
CLS=refine_final
CORHM=1
DCN_ZERO_BASE=0
BASE_PLANE=32

#ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE  --lr 0.01 --base_plane $BASE_PLANE --resume $CP --bs 1

#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE --lr 0.05 --base_plane $BASE_PLANE


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
STYLE=--show

ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE $STYLE --base_plane $BASE_PLANE 


