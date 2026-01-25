#%%
import torch
import pytorch3d
from pytorch3d.ops import knn_points


#%%
print(pytorch3d.__version__)
print(f"torch version: {torch.__version__}")
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)


#%%
p1 = torch.rand((1, 100, 3), device='cuda')
p2 = torch.rand((1, 200, 3), device='cuda')
knn = knn_points(p1, p2, K=1)
print("KNN distances shape:", knn.dists.shape)


#%%
