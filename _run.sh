# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/bev_strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/bev_strpoints_r50_fpn_1x_BK.py
#CONFIG=configs/strpoints/S3dProj_BevResNet_strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/S3dProj_BevResNet_strpoints_r50_fpn_1x_BK.py


wkdir=sTPV_r50_fpn_stanford2d_wa_bs7_lr10_LsW510R2P1N1_Rfiou743_Fpn44_Pbs1_Bp32_Fe
CP=./work_dirs/${wkdir}/best.pth
#CONFIG=./work_dirs/${wkdir}/_S3dProj_BevResNet_strpoints_r50_fpn_1x.py
CONFIG=./work_dirs/${wkdir}/_bev_strpoints_r50_fpn_1x.py
#CP=./checkpoints/beike/Apr16FineTuneApr12_Fpn44_Bp32.pth


LR=0.01
ROTATE=0
BASE_PLANE=32
BS=7
DATA_TYPES=cnx
FILTER_EDGES=1
CLS=adi
CLS=ad
CLS=a

#ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --lr $LR --base_plane $BASE_PLANE --bs $BS  --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES  --classes $CLS --resume $CP 

#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE   --lr $LR --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES  --classes $CLS  
#CLS=abcd
#--resume $CP 
#--bs $BS


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show

ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE   $STYLE --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES   --classes $CLS
