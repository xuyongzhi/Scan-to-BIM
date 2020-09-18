# Environment tested
- Nvidia driver 440.82
- CUDA 10.2
- pytorch 1.4.0
- python 3.7.6    

# Installation
1. Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.
2. Install detectron 2
```
# for CUDA 10.2:
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/index.html
```
Or refer to [Detectron 2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)


# Data generation
1. make dir: mmdetection/data/beike
2. 将已经merge好的 pcl 和　json 链接到:   
        mmdetection/data/beike/data/ply  
        mmdetection/data/beike/data/json  
3. 
```
cd mmdetection/beike_data_utils
python data_preprocess.py

在　data/beike/processed_512　中会生成一下文件：
all.txt  json  mean_std.txt  pcl_scopes  ply  relationImgs  relations  room_bboxes  test.txt  TopView_VerD  TopView_VerD_Imgs  train.txt
```
1. 配置　data_preprocess.py　中的　pool_num　可以设置多进程。 
2. scene_start=0, max_scene_num = 100 控制处理的数据范围。预处理最开始会将所有的scene排序，把第scene_start到scene_start+max_scene_num 放到　all.txt。后续所有的操作都只针对all.txt中的scene进行。
比如可以先跑： scene_start=0, max_scene_num = 100 
再跑： scene_start=100, max_scene_num = 100  (第100 到　200)
重复处理不会增加太多的时间，因为只检测文件是否存在。
再跑： scene_start=0, max_scene_num = 200  应该会很快。
3. json和ply 是链接。在training 过程中需要加载 json，　但不加载ply。
4. TopView_VerD_Imgs 和　relationImgs　仅用于验证生成效果，在训练是不需要。
5. train.txt 和　test.txt 每次都会随机打乱更新。
6. 如果某个scene 出错，应该是这个scene生成了　size 为0的文件，删掉后重跑应该可过。


# Training

./run.sh

# Show results
## shown gt pcl models
``` 
        cd ./beike_data_utils 
        python beike_utils.py

        In gen_gt_pcl_3d_models, show_3d=1
```

## Evaluation
```
        cd ./utils_dataset
        python graph_eval_utils.py

```

