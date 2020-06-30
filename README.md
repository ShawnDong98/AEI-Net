# AEI-Net

# Setup

安装requirements.txt依赖

> pip install -r requirements.txt

将训练数据集放在data下的train文件夹

# Run

## 训练

### 使用分布式单机多卡

> python -m torch.distributed.launch --nproc_per_node=num_gpu train_DDP.py

num_gpu为GPU数

需要在train_DDP.py文件中设置os.environ['CUDA_VISIBLE_DEVICES']， 表示可用的GPU

### 使用单卡


> python train.py


## 预测

> python inference.py --img1_path=path1 --img2_path=path2

path1和path2为交换人脸的路径， 也可以直接在inference.py中直接修改default

# Tips

所有训练和测试的图片都需要使用face_modules下的preprocess_images.py进行预处理