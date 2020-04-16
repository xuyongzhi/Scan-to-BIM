# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/pcl_dense3dfpn_strpoints_r50_fpn_1x.py
CONFIG=configs/strpoints/S3dProj_BevResNet_strpoints_r50_fpn_1x.py


wkdir=R50_fpn_beike_pcl_2d_wi_bs4_lr10_cnx_DaugR2P1N1_Rfiou743_Fpn44_Pbs1_Bp32_Vsz4Stem4-D90_0K_zX
CP=./work_dirs/${wkdir}/best.pth
#CONFIG=./work_dirs/${wkdir}/_pcl_dense3dfpn_strpoints_r50_fpn_1x.py
#CONFIG=./work_dirs/${wkdir}/_strpoints_r50_fpn_1x.py


LR=0.01
ROTATE=1
BASE_PLANE=32
BS=2
DATA_TYPES=cnx
FILTER_EDGES=1
#CLS=aid
CLS=a
#CLS=i

ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --lr $LR --base_plane $BASE_PLANE --bs $BS  --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES  --classes $CLS  
#--resume $CP 

#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE   --lr $LR --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES  --classes $CLS 
#--resume $CP 


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show

#ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE   $STYLE --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES   --classes $CLS
