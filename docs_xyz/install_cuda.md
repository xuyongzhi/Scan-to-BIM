# CUDA
Gcc 7.4 is not compatiable with cuda10.1 while building sponv. Use Ubuntu 6.5.0-2ubuntu1~18.04
- ref1: https://medium.com/repro-repo/install-cuda-10-1-and-cudnn-7-5-0-for-pytorch-on-ubuntu-18-04-lts-9b6124c44cc
- ref2: https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73


| GPU | Linux | NVIDIA Driver|CUDA | CUDNN |
|--|--|--|--|--|
|1080TI|Ubuntu 18  |390.116  | 9.0.176/9.2| |
|2080TI|Ubuntu 18  |430.50  | 10.1  | 7.6.5.33 |

## Clean
```
sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

## NVIDIA Driver
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-driver-430
reboot
nvidia-smi
```

## CUDA
```
https://developer.nvidia.com/cuda-toolkit-archive
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.105-418.39/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```
edit ~/.bashrc
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```
Check:
```
source ~/.bashrc
cd /usr/local/cuda-10.1/samples
sudo make
/usr/local/cuda-10.1/samples/bin/x86_64/linux/release/deviceQuery
```

## CUDNN
```
wget 3 debs
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.1_amd64.deb
cd /usr/src/cudnn_samples_v7/mnistCUDNN/
sudo make clean && sudo make
./mnistCUDNN
```
