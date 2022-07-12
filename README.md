# GCISG

This is an implementation of GCISG: Guided Causal Invariant Learning for Improved Syn-to-real Generalization. (Accepted to ECCV 2022)

<img width="80%" img src="https://user-images.githubusercontent.com/33364883/178862016-19276fc8-cd17-4ac6-9392-058b9cc9df6d.jpg"/>

## Environment Setup

Tested in a Python 3.8 environment in Linux with:

- Pytorch: 1.8.1
- [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning): 1.3.1

## Classification (VisDA17)

### Dataset Setup

Download [VisDA17 dataset from official website](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) and extract / place them as below.

```
📂 datasets
 ┣ 📂 visda17
 ┃ ┣ 📂 train
 ┃ ┃ 📂 validation
 ┗ ┗ 📂 test
```

### Train

```
python run.py
```

## Semantic Segmentation

TBD

## Object detection

TBD
