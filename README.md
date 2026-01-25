# Material_Generative_Model
## Related Works
1. [DeepCAD: A Deep Generative Network for Computer-Aided Design Models](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_DeepCAD_A_Deep_Generative_Network_for_Computer-Aided_Design_Models_ICCV_2021_paper.pdf)
2. [Learning Representations and Generative Models for 3D Point Clouds](https://proceedings.mlr.press/v80/achlioptas18a/achlioptas18a.pdf)
3. [Diffusion-CAD: Controllable Diffusion Model for Generating Computer-Aided Design Models](https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox/FMfcgzQbfVGvDcdlBjJsqqtbCRhCFmpr?projector=1&messagePartId=0.3)
4. [CAPP-GPT: A Large Multimodal Model via a Custom](https://www.techrxiv.org/doi/full/10.36227/techrxiv.174831482.26940243)
5. [GenCAD: Image-Conditioned Computer-Aided Design Generation](https://arxiv.org/abs/2409.16294)
## Implementation of 3D Point Clouds
1. [official implementation by tensorflow](https://github.com/optas/latent_3d_points/tree/master)
2. [pytorch implementation](https://github.com/cihanongun/Point-Cloud-Autoencoder/tree/master)---This implementation focuses on AE-based training.
## Install packages to run the code
[install pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

**Remark.** Ensure that the GPU version of pytorch3d is installed.

## Steps to generate 3D Point Clouds
1. Train an Autoencoder (AE), consisting of an encoder $En$ and a decoder $De$, and save the trained model. (done)
2. Train a Wasserstein GAN (WGAN), composed of a generator $G$ and a discriminator $D$, to learn the distribution of encoded 3D point clouds $En(X)$, where $X$ represents the original 3D point cloud data. The goal is for the distribution of $G(\epsilon)$ to match that of $En(X)$, where $\epsilon \sim \mathcal{N}(0, 1)$. Save the trained generator $G$. (done)
3. To generate a new 3D point cloud, sample a noise vector $\epsilon$ from the standard normal distribution $\mathcal{N}(0, 1)$, then compute $De(G(\epsilon))$. This output is a synthesized 3D point cloud.

**Remark.** [link to data and saved models](https://1drv.ms/f/c/394ed6f97bd226e4/Et0gs6rFg5BPjlRlgpiF25EBdAqEA4EAHQ53c8j5VKKg-Q)

## Tasks to complete
1. Create a virtual Python environment and install the necessary packages to run the code.  
2. Explore and analyze the Gcode dataset Slice-100K to gain a thorough understanding of its structure and contents.
3. Perform feature engineering using domain-specific knowledge, following a similar approach to Section 3.1.2 ("Network-Friendly Representation") in DeepCAD.

## GAI for AM
1. [Online distortion simulation using generative machine learning models](https://www.sciencedirect.com/science/article/pii/S2452414X24000074)
2. [A physics-guided deep generative model for predicting melt pool](https://link.springer.com/article/10.1007/s10845-024-02504-1)
## Dataset
1. Gcode database Slice-100K: A Multimodal Dataset for Extrusion-based 3D Printing  https://arxiv.org/abs/2407.04180   https://figshare.com/s/9d084ff84f3822d2bf17
2. LLM course: https://github.com/mlabonne/llm-course
   
