# Latent Space WGAN to Generate 3D POint Clouds

## [install pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

**Remark.** Ensure that the GPU version of pytorch3d is installed.

## Method
1. Train an Autoencoder (AE), consisting of an encoder $En$ and a decoder $D$, and save the trained model.
2. Train a Wasserstein GAN (WGAN), composed of a generator $G$ and a discriminator $D_e$, to learn the distribution of encoded 3D point clouds $E_n(X)$, where $X$ represents the original 3D point cloud data. The goal is to make the distribution of $G(\epsilon)$ match that of $E_n(X)$, where $\epsilon \sim \mathcal{N}(0, 1)$. Save the trained generator $G$.

**Remark**. To generate a new 3D point cloud, sample a noise vector $\epsilon$ from the standard normal distribution $\mathcal{N}(0, 1)$, then pass it through the generator and decoder to compute $D(G(\epsilon))$. The resulting output is a synthesized 3D point cloud.


