# FSMIS via MTPNet

![image](https://github.com/zmcheng9/GMRD/blob/main/overview.png)

### Abstract
Given the high annotation costs and ethical considerations associated with medical images, leveraging a limited number of annotated samples for Few-Shot Medical Image Segmentation (FSMIS) has become increasingly prevalent. However, existing models tend to focus on visible foreground support information, often overlooking extreme foreground-background imbalances. In addition, query images sometimes have slight different appearance compared to support images of the same category due to the differences in size as well as slicing angle, thus employing only support images to generate prototypes inevitably leads to matching bias.
To address these challenges, we present an innovative approach through learning a Multiple Twin-support Prototypes Network (MTPNet). Our approach includes the design of the Scale Consistent Sampling (SCS) module, which adaptively adjusts the foreground and background points within the support set, thereby balancing the influence of various structural elements in the image. Additionally, the Twin-support Prototypes Extraction (TPE) module facilitates the critical interaction between query and support features to extract twin-support prototypes. This module incorporates a Backtrace Interaction Filter (BIF) to eliminate erroneous interaction prototypes. Extensive experimental validation on three widely used medical image datasets demonstrates that our method surpasses current State-of-the-arts, showcasing its potential to address key limitations in FSMIS.

### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Data sets and pre-processing
Download:
1) **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

### Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train.sh` 

### Inference
Run `./script/evaluate.sh` 

### Citation
```
@ARTICLE{cheng2024few,
  author={Cheng, Ziming and Wang, Shidong and Xin, Tong and Zhou, Tao and Zhang, Haofeng and Shao, Ling},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Few-Shot Medical Image Segmentation via Generating Multiple Representative Descriptors}, 
  year={2024},
  volume={},
  number={},
  pages={1-1}
  doi={10.1109/TMI.2024.3358295}}
```
