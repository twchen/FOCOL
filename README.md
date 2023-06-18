# FOCOL
This repository contains the code for the paper "Improving Representation Learning for Session-based Recommendation".

## Requirements
The code has been tested on an environment with Python 3.9.7, PyTorch 1.9.1, Numpy 1.21.2, and DGL 0.7.1.

## Datasets
The datasets are taken from the previous work [Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation (AAAI'21)](https://arxiv.org/abs/2012.06852). You can download the datasets in [this GitHub repo](https://github.com/xiaxin1998/DHCN/) and extract them to the [datasets](./datasets) folder.

## Usage
To train FOCOL, run the following command:
```bash
python run.py expts/focol/focol.py --dataset-dir datasets/diginetica
```
which trains FOCOL on _diginetica_ with default hyperparameters.

You can see the detailed usage with the following command:
```bash
python run.py expts/focol/focol.py -h
```

## Citation
```
If you find the code useful, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/10020851):
@inproceedings{chen2022focol,
   title="Improving Representation Learning for Session-based Recommendation",
   author="Tianwen {Chen} and Raymond Chi-Wing {Wong}",
   booktitle="IEEE International Conference on Big Data",
   year="2022"
}
```
