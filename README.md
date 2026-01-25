# Latent Space WGAN to Generate 3D POint Clouds

## Install packages to run the code
2. [install pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

**Remark.** Ensure that the GPU version of pytorch3d is installed.

## Steps to generate 3D Point Clouds
1. Train an Autoencoder (AE), consisting of an encoder $En$ and a decoder $De$, and save the trained model. (done)
2. Train a Wasserstein GAN (WGAN), composed of a generator $G$ and a discriminator $D$, to learn the distribution of encoded 3D point clouds $En(X)$, where $X$ represents the original 3D point cloud data. The goal is for the distribution of $G(\epsilon)$ to match that of $En(X)$, where $\epsilon \sim \mathcal{N}(0, 1)$. Save the trained generator $G$. (done)
3. To generate a new 3D point cloud, sample a noise vector $\epsilon$ from the standard normal distribution $\mathcal{N}(0, 1)$, then compute $De(G(\epsilon))$. This output is a synthesized 3D point cloud.

**Remark.** [link to data and saved models](https://1drv.ms/f/c/394ed6f97bd226e4/Et0gs6rFg5BPjlRlgpiF25EBdAqEA4EAHQ53c8j5VKKg-Q)

