import torch

# La red es diminuta: varios hilos por operacion cuestan mas de lo que ahorran.
torch.set_num_threads(1)
