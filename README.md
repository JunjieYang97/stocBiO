# Efficient bilevel Optimizers stocBiO, ITD-BiO and FO-ITD-BiO.
Codes for ICML 2021 paper [Bilevel Optimization: Nonasymptotic Analysis and Faster Algorithms](https://arxiv.org/pdf/2010.07962.pdf) by Kaiyi Ji, Junjie Yang, and Yingbin Liang from The Ohio State University.

## stocBiO for hyperparameter optimization
Our hyperparameter optimization implementation is bulit on [HyperTorch](https://github.com/prolearner/hypertorch), where we propose stoc-BiO algorithm with better performance than other bilevel algorithms. Our code is tested on python3 and PyTorch1.8. 

The experiments based on 20 Newsgroup and MNIST datasets are in [l2reg_on_twentynews.py](https://github.com/JunjieYang97/stocBiO/blob/master/Hyperparameter-optimization/experimental/l2reg_on_twentynews.py) and [mnist_exp.py](https://github.com/JunjieYang97/stocBiO/blob/master/Hyperparameter-optimization/experimental/mnist_exp.py), respectively.

### How to run our code

We introduce some basic args meanning as follows.

#### Args meaning

+ `--alg`: Different algorithms we support.
+ `--hessian_q`: The number of Hessian vectors used to estimate.
+ `--training_size`: The number of samples used in training.
+ `--validation_size`: The number of samples used for validation.
+ `--batch_size`: Batch size for traning data.
+ `--epochs`: Outer epoch number for training.
+ `--iterations` or `--T`: Inner iteration number for training.
+ `--eta`: Hyperparameter $\eta$ for Hessian inverse approximation.
+ `--noise_rate`: The corruption rate for MNIST data.

To replicate empirical results under different datasets in our paper, please run the following commands:

#### stocBiO in MNIST with p=0.1

```python
python3 mnist_exp.py --alg stocBiO --batch_size 50 --noise_rate 0.1
```

#### stocBiO in MNIST with p=0.4

```python
python3 mnist_exp.py --alg stocBiO --batch_size 50 --noise_rate 0.4
```

#### stocBiO in 20 Newsgroup

```python
python3 l2reg_on_twentynews.py --alg stocBiO
```

#### AID-FP in MNIST with p=0.4

```python
python3 mnist_exp.py --alg AID-FP --batch_size 50 --noise_rate 0.4
```

#### AID-FP in 20 Newsgroup

```python
python3 l2reg_on_twentynews.py --alg AID-FP
```



## ITD-BiO and FO-ITD-BiO for meta-learning
Our meta-learning part is built on [learn2learn](https://github.com/learnables/learn2learn), where we implement the bilevel optimizer ITD-BiO and show that it converges faster than MAML and ANIL. Note that we also implement first-order ITD-BiO (FO-ITD-BiO) without computing the derivative of the inner-loop output with respect to feature parameters, i.e., removing all Jacobian and Hessian-vector calculations. It turns out that FO-ITD-BiO is even faster without sacrificing overall prediction accuracy.  

## Some experiment examples

In the following, we provide some experiments to demonstrate the better performance of the proposed stoc-BiO algorithm. 

We compare our algorithm to various hyperparameter baseline algorithms on 20 Newsgroup dataset:

<img src="Hyperparameter-optimization/results/test_loss_alg.png" width="350">

We evaluate the performance of our algorithm with respect to different batch sizes:

<img src="Hyperparameter-optimization/results/test_loss_batch.png" width="350">

The comparison results on MNIST dataset:

<img src="Hyperparameter-optimization/results/test_loss_mnist.png" width="350">

This repo is still under construction and any comment is welcome! 

## Citation

If this repo is useful for your research, please cite our paper:

```tex
@inproceedings{ji2021bilevel,
	author = {Ji, Kaiyi and Yang, Junjie and Liang, Yingbin},
	title = {Bilevel Optimization: Nonasymptotic Analysis and Faster Algorithms},
	booktitle={International Conference on Machine Learning (ICML)},
	year = {2021}}
```

