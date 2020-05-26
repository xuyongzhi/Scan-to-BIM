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
```

4. 验证生成的图片： mmdetection/data/beike/processed_512/TopView_VerD 
5. 将 mmdetection/data/beike/processed_512/train.txt 和 test.txt 换成官方的设置。 

# Training

./run.sh
