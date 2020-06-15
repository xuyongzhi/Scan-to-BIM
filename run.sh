# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/d_bev_strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/S3dProj_BevResNet_strpoints_r50_fpn_1x.py


wkdir=bTPV_r50_fpn_XYXYSin2WZ0Z1_Std__beike2d_wado_bs7_lr10_LsW510R2P1N1_Rfiou741_Fpn44_Pbs1_Bp32_Rel
#wkdir=test

#CONFIG=./work_dirs/${wkdir}/_S3dProj_BevResNet_strpoints_r50_fpn_1x.py
#CONFIG=./work_dirs/${wkdir}/_bev_strpoints_r50_fpn_1x.py

#CP=./work_dirs/${wkdir}/best.pth
CP=./checkpoints/beike/jun2_wado_bev.pth
#CP=./checkpoints/beike/May4_wd_Bev.pth
#CP=./checkpoints/sfd/15May_Pcl_abcdi_train_6as.pth


LR=0.0001
ROTATE=1
BASE_PLANE=32
BS=7
DATA_TYPES=cnx
FILTER_EDGES=0
REL=1

#CLS=abcdif
CLS=da
#CLS=r

# single gpu
ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --lr $LR --base_plane $BASE_PLANE --bs $BS  --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES  --classes $CLS --relation $REL  
#--resume $CP 


# Multi gpu
#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE   --lr $LR --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES  --classes $CLS   --relation $REL
#--resume $CP 


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show

#ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE   $STYLE --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES   --classes $CLS  --relation $REL
