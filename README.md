1、GPU环境
NVIDIA 460.80.02 CUDA 11.0

2、安装Anaconda
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.07-Linux-x86_64.sh
chmod +x Anaconda3-2020.07-Linux-x86_64.sh
./Anaconda3-2020.07-Linux-x86_64.sh

3、安装Python环境
conda create -n sloth python=3.7
conda activate sloth
conda install pytorch=1.6.0 torchvision cudatoolkit=10.2 -c pytorch
pip install mmcv-full
unzip -q mmdetection-2.5.0.zip
cd mmdetection-2.5.0
pip install -r requirements/build.txt
pip install -v -e .

4、训练模型
python tools/train.py shelf_configs/mask_rcnn_r50_fpn.py

5、模型推理
python inference.py test_img/
