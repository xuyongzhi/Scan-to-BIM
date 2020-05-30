# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/bev_strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/S3dProj_BevResNet_strpoints_r50_fpn_1x.py


wkdir=bTPV_r50_fpn_XYXYSin2__beike2d_wado_bs6_lr10_LsW510R2P1N1_Rfiou741_Fpn44_Pbs1_Bp32
wkdir=test
#CONFIG=./work_dirs/${wkdir}/_S3dProj_BevResNet_strpoints_r50_fpn_1x.py
#CONFIG=./work_dirs/${wkdir}/_bev_strpoints_r50_fpn_1x.py

#CP=./work_dirs/${wkdir}/best.pth
CP=./checkpoints/beike/May4_wd_Bev.pth
#CP=./checkpoints/sfd/15May_Pcl_abcdi_train_6as.pth


LR=0.01
ROTATE=1
BASE_PLANE=32
BS=5
DATA_TYPES=cnx
FILTER_EDGES=1
REL=0

#CLS=abcdif
CLS=ad
#CLS=a
#CLS=A

#CLS=acd
#CLS=bif

ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show

ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE   $STYLE --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES   --classes $CLS  --relation $REL