# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/pcl_dense3dfpn_strpoints_r50_fpn_1x.py


wkdir=TPV_r50_fpn_refine_final_beike2d_bs7_lr10_512_VerD_RA_Normrawstd_Rfiou743_Fpn34_Pbs1_Bp64_Fe-D90_0K_zX
CP=./work_dirs/${wkdir}/best.pth
#CONFIG=./work_dirs/${wkdir}/_pcl_dense3dfpn_strpoints_r50_fpn_1x.py
CONFIG=./work_dirs/${wkdir}/_strpoints_r50_fpn_1x.py


LR=0.01
ROTATE=1
BASE_PLANE=64
BS=1
DATA_TYPES=cnx
FILTER_EDGES=1

#ipython tools/train.py --  ${CONFIG} --rotate $ROTATE --lr $LR --base_plane $BASE_PLANE --bs $BS  --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES
#--resume $CP 

#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE   --lr $LR --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show

ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE   $STYLE --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES
