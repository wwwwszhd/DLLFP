import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import torch

train_file = sio.loadmat('train_random.mat')
train_x = train_file['one_hot_data']
train_y = train_file['label']

test_file = sio.loadmat('test_random.mat')
test_x = test_file['one_hot_data']
test_y = test_file['label']

train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.int64).reshape(1088)

test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.int64).reshape(120)

torch.save(train_x, 'AVP_random_train_x.pt')
torch.save(test_x, 'AVP_random_test_x.pt')
torch.save(train_y, 'AVP_random_train_y.pt')
torch.save(test_y, 'AVP_random_test_y.pt')