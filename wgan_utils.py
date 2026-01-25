

import torch

@torch.no_grad()
def generate_sample(generator, ae, batch_size, device, latent_dim=128):
    noise = torch.randn(batch_size, latent_dim, device=device)
    z_gen = generator(noise)
    x = ae.decoder(z_gen).cpu()
    return x

def generate_embedding(generator, batch_size, device, latent_dim=128):
    noise = torch.randn(batch_size, latent_dim, device=device)
    return generator(noise)

