# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

#CONFIG=configs/strpoints/S3dProj_BevResNet_strpoints_r50_fpn_1x.py


wkdir=bTPV_r50_fpn_XYXYSin2WZ0Z1_Std__beike2d_ro_bs7_lr1_LsW510R2P1N1_Rfiou832_Fpn44_Pbs1_Bp32
#wkdir=test

#CONFIG=./work_dirs/${wkdir}/_S3dProj_BevResNet_strpoints_r50_fpn_1x.py
#CONFIG=./work_dirs/${wkdir}/_bev_strpoints_r50_fpn_1x.py

#CP=./work_dirs/${wkdir}/best.pth
#CP=./checkpoints/beike/jun17_wd_bev_L.pth
#CP=./checkpoints/beike/jun18_r_bev_L.pth
#CP=./checkpoints/sfd/15May_Pcl_abcdi_train_6as.pth


LR=0.01
LR=0.001
ROTATE=1
BASE_PLANE=32
BS=7
DATA_TYPES=cnx
FILTER_EDGES=0


CLS=r
REL=0

ROTATE=0

wkdir=bTPV_r50_fpn_XYXYSin2WZ0Z1_Std__beike2d_ro_bs7_lr1_LsW510R2P1N1_Rfiou832_Fpn44_Pbs1_Bp32
CP=./work_dirs/${wkdir}/latest.pth
CONFIG=configs/strpoints/bev_strpoints_r50_fpn_1x_r.py
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'

#ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE   $STYLE --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES   --classes $CLS  --relation $REL 


CLS=a

wkdir=bTPV_r50_fpn_XYXYSin2_beike2d_wa_bs7_lr1_LsW510R2P1N1_Rfiou631_Fpn44_Pbs1_Bp32
CP=./work_dirs/${wkdir}/latest.pth
CONFIG=configs/strpoints/bev_strpoints_r50_fpn_1x.py
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'

ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE   $STYLE --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES   --classes $CLS  --relation $REL 
