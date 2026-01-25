#
from box import Box
from pathlib import Path


def Config(load_ae=True, save_results = True):
    '''
    This function is to define hyper-parameters
    return is a python box object
    '''
    cfg = Box(default_box=True)
    cfg.device = "cuda"
    cfg.load_ae = load_ae

    cfg.base_dir = Path.home()/"onedrive_local/research/3D_Points_Clouds"
    cfg.data_dir = cfg.data_dir / "data"
    cfg.dataset = "chair_set.npy"
    cfg.model_dir = cfg.data_dir / "model"
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_folder = "output/"
    cfg.output_folder.mkdir(parents=True, exist_ok=True)

    # hyper-parameters to train AE
    cfg.batch_size = 32
    cfg.ae_epoch = 1001
    cfg.latent_size = 128    # bottleneck size of the AE
    cfg.save_results = save_results

    # hyper-parameters to train WGAN
    cfg.gan_epoch = 100
    cfg.gan_batch = 64
    cfg.lambda_gp = 10

    return cfg
