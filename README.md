# A Hybrid Neural Coding Approach for Pattern Recognition with Spiking Neural Networks 
This repository contains the implementation of the paper [A Hybrid Neural Coding Approach for Pattern Recognition with Spiking Neural Networks](https://ieeexplore.ieee.org/document/10347028/metrics#metrics) published at **IEEE Transactions on Pattern Analysis Machine Intelligence (TPAMI)**. 

## Dataset
CIFAR-10

## Dependencies
- python 3.9.18
- Pytorch 1.13.0
- [Spikingjelly 0.0.0.12](https://github.com/fangwei123456/spikingjelly).

## Reducibility

* Pre-train the ANN.
  ```
  python ./CIFAR10/ANN_baseline/cifar10_vgg16_base_model.py
  python ./CIFAR10/ANN_baseline/cifar10_resNet20_base_model.py
  ```
* Training Hybrid Coding framework. 
  ```
  python ./CIFAR10/Hybrid_coding/cifar10_main_vgg16.py
  python ./CIFAR10/Hybrid_coding/cifar10_main_res20.py
  ```

## Citation 
```tex
@article{chen2024hybrid,
  title={A hybrid neural coding approach for pattern recognition with spiking neural networks},
  author={Chen, Xinyi and Yang, Qu and Wu, Jibin and Li, Haizhou and Tan, Kay Chen},
  journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence},
  volume={46},
  number={05},
  pages={3064--3078},
  year={2024},
  publisher={IEEE Computer Society}
}
```
