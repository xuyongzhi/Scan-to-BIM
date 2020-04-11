
# RES

## 11 Apr
e52b9d6fc7df1a8ae341b269afffd89250e3c273
TPV_r50_fpn_refine_final_beike2d_bs7_lr10_512_VerD_RA_Normrawstd_Rfiou743_Fpn34_Pbs1_Bp64_Fe-D90_0K_zX
|config | eval set | corner prec-recall| line prec-recall |
|-|-|-|-|
|Bp32_Vsz4Stem4| train, line_ave, no rotation    |  0.933 - 0.906  | 0.905 - 0.788  |
|              | test                          |  0.877 - 0.822 | 0.77 - 0.639  |


## 10 Apr

## RES 11 Apr Img
622cbcd0fa5270dc5ee14306a49f85d26bd57d64
TPV_r50_fpn_refine_final_beike2d_bs6_lr10_512_VerD_RA_Normrawstd_Rfiou743_Fpn34_Pbs1_Bp64-D90_0K_zX
Empty gt edges not filtered
|config | eval set | corner prec-recall| line prec-recall |
|-|-|-|-|
|Bp32_Vsz4Stem4| train, line_ave, no rotation    |  0.942 - 0.944  | 0.916 - 0.845  |
|              | test                          |  0.825 - 0.82 | 0.734 - 0.596  |

## RES 10 Apr Pcl
cc35d1814317a79f14aeed300b431cc1422cfd6d
R50_fpn_refine_final_beike_pcl_2d_bs5_lr10_cnx_Daug_Rfiou743_Fpn34_Pbs1_Bp64_Vsz4Stem4
Empty gt edges filtered
|config | eval set | corner prec-recall| line prec-recall |
|-|-|-|-|
|Bp32_Vsz4Stem4| train, line_ave, no rotation    |  0.897 - 0.844  | 0.833 - 0.685  |
|              | test                          |  0.827 - 0.768 | 0.742 - 0.582  |

## RES Img 8/Mar
```
8/Mar T90_r50_fpn_lscope_istopleft_refine_final_512_VerD_bs6_lr10_RA_Normrawstd_ChmR2P1N1_Rfiou743_Fpn34_Pbs1
```
|config | eval set | corner prec-recall| line prec-recall |
|-|-|-|-|
|Rfiou743_Fpn34_Pbs1 | test, composite, no rotation    | 0.828 - 0.82  |0.694 - 0.638  |
|Rfiou743_Fpn34_Pbs1 | test, line_ave, no rotation     | 0.842 - 0.836 |0.746 - 0.649  |
|Rfiou743_Fpn34_Pbs1 | test, 1 stage, no rotation | 0.715 - 0.833 |0.637 - 0.545  |
| | | |
|Rfiou743_Fpn34_Pbs1 | train, composite, no rotation    | 0.909 - 0.93  |  0.826 - 0.827 |
|Rfiou743_Fpn34_Pbs1 | train, line_ave, no rotation     | 0.932 - 0.926 | 0.908 - 0.822 |
|Rfiou743_Fpn34_Pbs1 | train, 1 stage, no rotation | 0.814 - 0.927 | 0.801 - 0.722  |


|config | eval set | corner prec-recall| line prec-recall |
|-|-|-|-|
| Rfiou743_Fpn34_Pbs2 | test, line_ave, no rotation     |  0.837 - 0.826 | 0.754 - 0.628  |
| Rfiou743_Fpn34_Pbs2 | train, line_ave, no rotation     | 0.918 - 0.91 | 0.891 - 0.786 |
| | | |
| Rfiou743_Fpn45_Pbs2 | test, line_ave, no rotation     |  0.821 - 0.813 | 0.748 - 0.622  |
| Rfiou743_Fpn45_Pbs2 | train, line_ave, no rotation     | 0.924 - 0.924 | 0.892 - 0.805 |

## RES Pcl 9 Img
|config | eval set | corner prec-recall| line prec-recall |
|-|-|-|-|
|Bp32_Vsz4Stem4| train, line_ave, no rotation    |  0.96 - 0.921  | 0.933 - 0.825  |
|              | test                          |  0.813 - 0.776 | 0.707 - 0.598  |
## RES Pcl 9 Apr
 Rfiou743_Fpn34_Pbs1
|config | eval set | corner prec-recall| line prec-recall |
|-|-|-|-|
|Bp32_Vsz4Stem4| train, line_ave, no rotation    | 0.867 - 0.807  | 0.8 - 0.614  |
|              | test                            |  0.824 - 0.764 | 0.721 - 0.533  |

