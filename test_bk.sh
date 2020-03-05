export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py


ROTATE=1
CLS=refine_final
CORHM=1
DCN_ZERO_BASE=0

wdir=T90_r50_fpn_lscope_istopleft_refine_final_512_VerD_bs5_lr10_RA_Normrawstd_ChmR2P1N1
CHECKPOINT=work_dirs/${wdir}/best.pth

STYLE='--show'
STYLE='--out ./work_dirs/'${wdir}'/detection.pickle --eval bbox'

ipython tools/test.py --  ${CONFIG} $CHECKPOINT --rotate $ROTATE --cls $CLS --corhm $CORHM --dcn_zero_base $DCN_ZERO_BASE $STYLE


