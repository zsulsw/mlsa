<p align="center"><img src="mlsa.png" width="500"\></p>

# Machine Learning-based Second-order Analysis of Beam-columns through Physics-Informed Neural Networks
The second-order analysis of slender steel members can be challenging, particularly when large deflections are involved. This research introduces a novel Machine Learning-based Structural Analysis (MLSA) method for the second-order analysis of beam-columns. This method presents a promising alternative to prevailing solutions that rely on oversimplified analytical equations or traditional finite-element-based methods.

The effectiveness of conventional machine learning methods heavily depends on the quality and quantity of the provided data. However, such data are often scarce and costly to obtain in structural engineering practices. To address this issue, we employ a new and explainable machine learning-based method called Physics-informed Neural Networks (PINN). This method uses physical information to guide the learning process, creating a self-supervised learning procedure. This approach makes it possible to train the neural network with few or even no predefined datasets, achieving an accurate approximation.

This research extends the PINN method to the problems of second-order analysis of steel beam-columns. The source code for the PINN program used in this paper is available on this GitHub page. We encourage readers to explore the code to gain a deeper understanding of the implementation details.

If you find our research and the provided source code useful, please consider citing our paper in your work.

## Requirements

- numpy~=1.24.2
- matplotlib~=3.7.0
- torch~=1.13.0
- tqdm~=4.65.0

## Citation

If the source codes are useful, please cite the paper [Click Here](http://dx.doi.org/10.18057/IJASC.2023.19.4.10).
```bibtex
@article{Chen-Liang-2023,
author = {Liang Chen, Hao-Yi Zhang, Si-Wei Liu and Siu-Lai Chan},
doi = {10.18057/IJASC.2023.19.4.10},
issn = {1816-112X},
journal = {Advanced Steel Construction},
pages = {411-420},
title = {{Second-order Analysis of Beam-columns by Machine Learning-based Structural Analysis through Physics-Informed Neural Networks}},
url = {http://dx.doi.org/10.18057/IJASC.2023.19.4.10},
volume = {19},
year = {2023}
}
```
