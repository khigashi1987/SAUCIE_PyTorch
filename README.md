# SAUCIE_PyTorch
PyTorch Implementation of SAUCIE Algorithm

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
* The resolution parameter in the cluster merging process (merge_k_nearest) can be adjusted.
* Modified Maximum Mean Discrepancy calculation. Multiple batch corrections can be learned simultaneously.