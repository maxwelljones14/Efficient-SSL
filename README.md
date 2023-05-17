# Efficient-SSL
Implementation for Efficient SSL paper Accepted to UAI 2023 (Arxiv pending)

We provide implementations for Algorithm 3 (Approx Feedback Set) using both HarmonicApprox (Algorithm 2 in the main body) and NonParametricApprox (Algorithm 2 in the appendix) as subroutines

For basic functionality to generate a set of plots using Approx Feedback Set (here we use PCA as preprocessing and kNN graph of size 6): 

```
python harmonic_approx.py --CG_steps=20 --num_experiments 3 --kNN 6 --PCA --dataset MNIST
```

Run 

```
run python gaussian_kernal_intervals.py --h
```
 for full list of arguments to pass in

For basic functionality to generate a set of plots using Non-Parametric Feedback Set: 

```
python non_parametric_approx.py --CG_steps=20 --PCA --kNN 6 --PCA --dataset FashionMNIST
```
