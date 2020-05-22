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
1. link Beike raw data to /data/beike/raws
2. python data_preprocess.py

# Training

./run.sh
