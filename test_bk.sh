#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py
CONFIG=configs/strpoints/T1_strpoints_r50_fpn_1x.py

wdir=T90_r50_fpn_lscope_istopleft_512_VerD_A_bs6_lr10_RA_Normrawstd
wdir=T1_r50_fpn_lscope_istopleft_512_VerD_A_bs1_lr10_RA_Normrawstd
CHECKPOINT=work_dirs/${wdir}/best.pth

ipython tools/test.py --  ${CONFIG} $CHECKPOINT  --rotate 1 --bs 1 --show

