## 🛠️ Getting Started

1. Clone repo
   
   ```bash
   git clone https://github.com/amai-gsu/Deformba.git
   cd Deformba
   ```
2. Create and activate a new conda environment
   
   ```bash
   conda create -n Deformba python=3.10
   conda activate Deformba
   ```
3. Install dependent packages
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   cd models/selective_scan && pip install .
   cd models/ops_dcnv3
   sh ./make.sh
   ```


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
  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=20000 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp
  ```
  
  To evaluate the performance with pre-trained weights:
  
  ```bash
  cd classification 
  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=20000 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp --pretrained </path/of/checkpoint> --eval
  ```
  To test the throughput of model:
  
  ```bash
  cd classification/models 
  python3 benchmark.py --batch-size 128 --model Deformba
  ```



