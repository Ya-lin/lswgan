# Latent Space WGAN to Generate 3D POint Clouds

## Requirements
The following software was installed on Ubuntu 24.04.3 LTS.
* Python 3.9.23
* CUDA 13.0
* numpy 1.26.4
* matplotlib 3.9.2
* torch 2.0.1+cu117
* pytorch-cuda 11.8
* pytorch3d 0.7.4

**Remark.** Ensure that the GPU-enabled version of PyTorch3D is installed. Please refer to the [installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for detailed instructions. 

## Method
1. Train an Autoencoder (AE), consisting of an encoder $E_n$ and a decoder $D_e$, and save the trained model. The reconstruction loss is defined as the Chamfer Distance between the input point set (X) and its reconstruction set $\hat{X}$, i.e., $L_{rec}(X,\hat{X})=\sum_{x\in X}\min_{\hat{x}\in\hat{X}}(x-\hat{x})^2+\sum_{\hat{x}\in\hat{X}}\min_{x\in X}(\hat{x}-x)^2$.
2. Train a Wasserstein GAN (WGAN), composed of a generator $G$ and a discriminator $D$, to learn the distribution of encoded 3D point clouds $E_n(X)$, where $X$ represents the original 3D point cloud data. The goal is to make the distribution of $G(\epsilon)$ match that of $E_n(X)$, where $\epsilon \sim \mathcal{N}(0, 1)$. Save the trained generator $G$.

**Remark**. To generate a new 3D point cloud, sample a noise vector $\epsilon$ from the standard normal distribution $\mathcal{N}(0, 1)$, then pass it through the generator and decoder to compute $D_e(G(\epsilon))$. The resulting output is a synthesized 3D point cloud.

## Steps to run
1. python train_ae.py
2. python train_wgan.py
3. python test_model.py
