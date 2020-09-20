# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/bev_strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/S3dProj_BevResNet_strpoints_r50_fpn_1x.py


wkdir=TrL18K_S_July3_PaperUsed/bTPV_r50_fpn_XYXYSin2_beike2d_wa_bs7_lr10_LsW510R2P1N1_Rfiou631_Fpn44_Pbs1_Bp32_Rel

#CONFIG=./work_dirs/${wkdir}/_S3dProj_BevResNet_strpoints_r50_fpn_1x.py
CONFIG=./work_dirs/${wkdir}/_bev_strpoints_r50_fpn_1x.py

CP=./work_dirs/${wkdir}/models_w/e1000_l1d01.pth
CP=./work_dirs/${wkdir}/models_w/e300_l1d88.pth
#CP=./work_dirs/${wkdir}/models_w/e400_l1d77.pth
#CP=./checkpoints/beike/jun17_wd_bev_L.pth
#CP=./checkpoints/beike/jun18_r_bev_L.pth


LR=0.01
ROTATE=1
BASE_PLANE=32
BS=7
DATA_TYPES=cnx
FILTER_EDGES=0

#CLS=abcdif

CLS=a
REL=1


# single gpu
#CLS=r
#REL=0
#ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --lr $LR --base_plane $BASE_PLANE --bs $BS  --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES  --classes $CLS --relation $REL  
#--resume $CP 

#ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --lr $LR --base_plane $BASE_PLANE --bs $BS  --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES  --classes $CLS --relation $REL  


# Multi gpu
#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE   --lr $LR --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES  --classes $CLS   --relation $REL
#--resume $CP 


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show
ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE   $STYLE --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES   --classes $CLS  --relation $REL 





