# xyz

#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1

CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/bev_sparse_strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/pcl_strpoints_r50_fpn_1x.py
#CONFIG=configs/strpoints/pcl_dense3dfpn_strpoints_r50_fpn_1x.py


wkdir=TPV_r50_fpn_refine_final_beike2d_bs5_lr10_512_VerD_RA_Normrawstd_ChmR2P1N1_Rfiou743_Fpn34_Pbs1_Bp64
CP=./work_dirs/${wkdir}/best.pth
#CONFIG=./work_dirs/${wkdir}/_pcl_strpoints_r50_fpn_1x.py


LR=0.01
ROTATE=1
CORHM=0
BASE_PLANE=64
BS=5
DATA_TYPES=cnx
FILTER_EDGES=1

ipython tools/train.py --  ${CONFIG} --rotate $ROTATE  --corhm $CORHM   --lr $LR --base_plane $BASE_PLANE --bs $BS  --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES
#--resume $CP 

#./tools/dist_train.sh ${CONFIG} 2 --rotate $ROTATE  --corhm $CORHM  --lr $LR --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES


ROTATE=0
STYLE='--out ./work_dirs/'${wkdir}'/detection.pickle --eval bbox'
#STYLE=--show

#ipython tools/test.py --  ${CONFIG} $CP --rotate $ROTATE  --corhm $CORHM  $STYLE --base_plane $BASE_PLANE   --data_types $DATA_TYPES  --filter_edges $FILTER_EDGES
