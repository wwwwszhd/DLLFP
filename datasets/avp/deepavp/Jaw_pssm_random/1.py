import numpy as np
import scipy.io as sio
from sklearn import model_selection

def load_train_data():
    filename = "./data/train.mat"
    file = sio.loadmat(filename)
    data = file['one_hot_data']
    sequence_length = file['sequence_length']
    label = file['label']

    return data, sequence_length, label
data, sequence_length, label = load_train_data()
print(data.shape)