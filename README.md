# SAUCIE_PyTorch
PyTorch Implementation of [SAUCIE](https://github.com/KrishnaswamyLab/SAUCIE) Algorithm

***For educational (or just enjoying) purposes only. Not fully tested for real dataset.***

## Requirements
```
numpy 1.16.2
pandas 0.24.1
pytorch 1.3.0
```

## Difference from the original implementation.
* Tensorflow old version => PyTorch
* Bug? fixes.
* Adjusted the impractical parameters (e.g. sigmas in RBF kernel function).
* Added the resolution parameter in the cluster merging process (merge_k_nearest).
* Modified Maximum Mean Discrepancy calculation. Multiple batch corrections can be learned simultaneously.

## Usage
See [notebook](https://khigashi1987.github.io/SAUCIE_PyTorch/)