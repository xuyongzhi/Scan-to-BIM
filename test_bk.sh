#export CUDA_VISIBLE_DEVICES=1
CONFIG=configs/strpoints/strpoints_r50_fpn_1x.py

wdir=T90_r50_fpn_lscope_istopleft_refine_512_VerD_bs3_lr10_RA_Normrawstd_DcnZb_Chm
CHECKPOINT=work_dirs/${wdir}/best.pth

ipython tools/test.py --  ${CONFIG} $CHECKPOINT  --rotate 1 --bs 1 --cls refine_final --corhm 1 --show --dcn_zero_base 1


