


## 🛠️ Getting Started

3. Install dependent packages
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   cd models/selective_scan && pip install .
   cd models/ops_dcnv3
   sh ./make.sh
   ```
   ```
   conda env create -f env.yml
   conda activate mamba-2d
   python -m pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   cd models/selective_scan && python -m pip install -v -e . --no-build-isolation
   error case solution 
   {    # 1) 强制 CUDA_HOME 指到 conda env（保证 $CUDA_HOME/bin/nvcc 存在）
        export CUDA_HOME="$CONDA_PREFIX"
        export PATH="$CUDA_HOME/bin:$PATH"

        # 2) 把 thrust 所在的 include 加入头文件搜索路径（不改源码）
        export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
        export C_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$C_INCLUDE_PATH"
        export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH"

        # 3) 清理缓存/中间产物
        rm -rf build *.egg-info ~/.cache/torch_extensions

        # 4) 重装
        python -m pip install -v -e . --no-build-isolation
  }
   cd models/ops_dcnv3
   sh ./make.sh
   ```

## ✨ Pre-trained Models


<summary> ImageNet-1k Image Classification </summary>
<br>




## 📚 Data Preparation

* ImageNet is an image database organized according to the WordNet hierarchy. Download and extract ImageNet train and val images from http://image-net.org/. Organize the data into the following directory structure:
  
  ```
  imagenet/
  ├── train/
  │   ├── n01440764/  (Example synset ID)
  │   │   ├── image1.JPEG
  │   │   ├── image2.JPEG
  │   │   └── ...
  │   ├── n01443537/  (Another synset ID)
  │   │   └── ...
  │   └── ...
  └── val/
      ├── n01440764/  (Example synset ID)
      │   ├── image1.JPEG
      │   └── ...
      └── ...
  ```
* COCO is a large-scale object detection, segmentation, and captioning dataset. Please visit http://cocodataset.org/ for more information, including for the data, paper, and tutorials. [COCO API](https://github.com/cocodataset/cocoapi) also provides a concise and efficient way to process the data.
* ADE20K is composed of more than 27K images from the SUN and Places databases. Please visit https://ade20k.csail.mit.edu/ for more information and see the [GitHub Repository](https://github.com/CSAILVision/ADE20K) for an overview of how to access and explore ADE20K.

## 🚀 Quick Start

* **Image Classification**
  
  To train Deformba models for classification on ImageNet, use the following commands for different configurations:
  
  ```bash
  cd classification 
  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp
  ```
  
  To evaluate the performance with pre-trained weights:
  
  ```bash
  cd classification 
  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp --pretrained </path/of/checkpoint>
  ```
  To test the throughput of model:
  
  ```bash
  cd classification/models 
  python3 benchmark.py --batch-size 128 --model Deformba_T
  ```



