#%%
import warnings
warnings.filterwarnings('ignore')


#%%
import yaml
import torch
import numpy as np
from box import Box
from pathlib import Path
from dataloaders import GetDataLoaders
from wgan_utils import generate_sample
from utils import set_seed, plotPCbatch
from models import PointCloudAE, Generator


#%%
# with open("config.yaml", "r") as file:
#     args = yaml.safe_load(file)
# args = Box(args)
# print(args)

# args.data_dir = Path.home()/args.data_dir
# args.model_dir = Path.home()/args.model_dir

# data_file = args.data_dir/args.dataset
# pc_array = np.load(data_file)
# print(f"data shape: {pc_array.shape}.")

# point_size = pc_array.shape[1]
# print(f"number of points in one 3D cloud: {point_size}.")
# loader, _ = GetDataLoaders(npArray=pc_array, 
#                            batch_size=args.batch_size, 
#                            train_set_percentage=1.0)


#%% 
if __name__ == "__main__":



    
    # load saved ae
    ae = PointCloudAE(point_size, args.latent_size).to(args.device)
    checkpoint = torch.load(args.model_dir/"ae.pt", weights_only=True)
    ae.load_state_dict(checkpoint)
    ae.eval()
    print("Successfully load ae!")

    # load saved generator
    generator = Generator().to(args.device)
    checkpoint = torch.load(args.model_dir/"wgan_g.pt", weights_only=True)
    generator.load_state_dict(checkpoint)
    generator.eval()
    print("Successfully load wgan generator!")

    #display generated samples
    sample1 = generate_sample(generator, ae, args.batch_size, args.device, latent_dim=128)
    sample2 = generate_sample(generator, ae, args.batch_size, args.device, latent_dim=128)
    plotPCbatch(sample1, sample2)


#%%