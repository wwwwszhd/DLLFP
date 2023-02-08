import scipy.io as sio
import torch

matrix = sio.loadmat('pssm.mat')['pssm']
matrix = torch.tensor(matrix, dtype=torch.float32)
torch.save(matrix, 'blosum_matrix.pt')
