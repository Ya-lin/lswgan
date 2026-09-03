#%%
import torch
import numpy as np
from box import Box
from torch import optim
from config import Config
from utils import set_seed
from torch.autograd import grad
from matplotlib import pyplot as plt
from dataloaders import GetDataLoaders
from models import PointCloudAE, Generator, Discriminator

import warnings
warnings.filterwarnings('ignore')


#%%
@torch.no_grad()
def generate_true_embedding(x, ae, device):
    x = x.to(device)
    true_embedding = ae.encoder(x.permute(0,2,1))
    return true_embedding

def generate_fake_embedding(generator, batch_size, device, latent_dim=128):
    noise = torch.randn(batch_size, latent_dim, device=device)
    return generator(noise)

@torch.no_grad()
def generate_sample(generator, ae, batch_size, device, latent_dim=128):
    z_gen = generate_fake_embedding(generator, batch_size, device, latent_dim)
    x = ae.decoder(z_gen).cpu()
    return x


#%%
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):

    # Compute gradient penalty for WGAN-GP, works with arbitrary input dimensions.
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, *[1] * (real_samples.dim() - 1), device=device)

    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    fake = torch.ones_like(d_interpolates, device=device)

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten each sample's gradients: (B, D1, D2, ...) -> (B, -1)
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


# training loop
def training_gan(epochs, latent_size, gan_batch, lambda_gp, device):

    print("start to train wgan")
    
    losses = Box(g=[],d=[])
    for epoch in range(epochs):
        dis_batch_losses = []
        gen_batch_losses = []
        for i, x in enumerate(loader):
            true_embedding = generate_true_embedding(x, ae, device)
            fake_embedding = generate_fake_embedding(gan_g, gan_batch, device, latent_size)
        
            # update discriminator
            d_real = gan_d(true_embedding)
            d_fake = gan_d(fake_embedding.detach())
            gp = compute_gradient_penalty(gan_d, true_embedding, fake_embedding.detach(), device)
            loss_d = -torch.mean(d_real) + torch.mean(d_fake) + lambda_gp * gp
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            dis_batch_losses.append(loss_d.item())

            # every 5 batches to update the generator
            if i % 5 == 0:
                loss_g = -torch.mean(gan_d(fake_embedding))
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                gen_batch_losses.append(loss_g.item())
        
        dis_avg_loss = np.mean(dis_batch_losses)
        gen_avg_loss = np.mean(gen_batch_losses)
        print(f"epoch {epoch+1}/{epochs} | loss_d: {dis_avg_loss:.4f} | loss_g: {gen_avg_loss:.4f}")

        losses.d.append(dis_avg_loss)
        losses.g.append(gen_avg_loss)

    return losses


#%% 
if __name__ == "__main__":

    # set random seed to make results reproducible 
    set_seed()

    # load config and data
    args = Config()
    print(args)

    data_file = args.data_dir/args.dataset
    pc_array = np.load(data_file)
    print(f"data shape: {pc_array.shape}.")

    # Assuming all models have the same size,
    point_size = pc_array.shape[1]
    print(f"number of points in one 3D cloud: {point_size}.")

    # no test dataset as the whole dataset is used to train WGAN
    loader, _ = GetDataLoaders(npArray=pc_array, 
                               batch_size=args.batch_size,
                               train_set_percentage=1.0)

    # load ae model
    ae = PointCloudAE(point_size, args.latent_size).to(args.device)
    checkpoint = torch.load(args.model_dir/"ae.pt", weights_only=True)
    ae.load_state_dict(checkpoint)
    ae.eval()
    print("Successfully load ae!")


    # train WGAN
    loader, _ = GetDataLoaders(npArray=pc_array, 
                           batch_size=args.gan_batch, 
                           train_set_percentage=1.0)
    
    gan_g = Generator().to(args.device)
    opt_g = optim.AdamW(gan_g.parameters(), lr=1e-4, betas=(0.5, 0.9))
    gan_d = Discriminator().to(args.device)
    opt_d = optim.AdamW(gan_d.parameters(), lr=1e-4, betas=(0.5, 0.9))

    losses = training_gan(args.gan_epoch, args.latent_size, args.gan_batch, args.lambda_gp, args.device)

    # display training loss
    fig = plt.figure()
    plt.plot(losses.g, label="generator loss")
    plt.plot(losses.d, label="discriminator loss")
    plt.legend()
    plt.show()
    fig.savefig(args.wgan_output/"loss.png")
    plt.close()

    # save model
    torch.save(gan_g.state_dict(), args.model_dir/"wgan_g.pt")
    torch.save(gan_d.state_dict(), args.model_dir/"wgan_d.pt")


#%%
