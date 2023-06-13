# Efficient-SSL
    Official implementation of [Efficiently Learning the Graph for Semi-supervised Learning](https://arxiv.org/abs/2306.07098)

We provide implementations for Algorithm 3 (Approx Feedback Set) using both HarmonicApprox (Algorithm 2 in the main body) and NonParametricApprox (Algorithm 1 in the appendix) as subroutines. If you already have the MNIST, FashionMNIST, or USPS pytorch datasets downloaded, let ROOT_PATH be the path to these datasets. If not, then set ROOT_PATH to the location where these datasets should be downloaded/stored upon use (default value is "/datasets/").

For basic functionality to generate a set of plots using Approx Feedback Set (here we use PCA as preprocessing and kNN graph of size 6): 

```
python harmonic_approx.py --CG_steps=20 --num_experiments 3 --kNN 6 --PCA --dataset MNIST --root_path ROOT_PATH
```

Run 

```
python harmonic_approx.py --h
```
 for full list of arguments to pass in.

For basic functionality to generate a set of plots using Non-Parametric Feedback Set: 

```
python non_parametric_approx.py --CG_steps=20 --PCA --kNN 6 --PCA --dataset FashionMNIST
```
NOTE: when running the two functions, it is ideal to have other tabs closed, as having many chrome tabs/other processes open greatly slows down runtime. 