# MTGS:Multi-task learning for concurrent grading diagnosis and semi-supervised segmentation of honeycomb lung in CT images
by Bingqian Yang,Xiufang Feng,Yunyun Dong
## Introduction
This repository is the Pytorch implementation of "Multi-task learning for concurrent grading diagnosis and semisupervised segmentation of honeycomb lung in CT images"
## Requirements
We implemented our experiment on the computer system of Taiyuan University of Technology. The specific configuration is as follows:

* Centos 7.4
* RTX Nvidia 3090 24G
  
Some important required packages include:

* CUDA 11.6
* Pytorch == 1.12.0
* Python == 3.9
* Some basic python packages such as Numpy, Scikit-image,  Scipy ......

## Usage
1. Download the Left Atrium dataset in [Google drive](). Put the data in './data/' folder
2. Train the model
```
cd code
python train_mynetwork.py
```
3. Test the model
```
cd code
python test_mynetwork.py
```

## Acknowledgement
Part of the code is revised from the UA-MT.

We thank Dr. Lequan Yu for their elegant and efficient code base.

