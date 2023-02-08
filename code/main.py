import numpy as np
import scipy.io as sio
import os
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt

import loss_for_sequence
from loss_for_sequence import *
from model import BiLSTM_CNN
from train_eval_test import train, eval, test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
peptide_function = 'ACP'
# sequence_length = 183
# batch_size = 128
num_epoch = 400
learning_rate = 0.01
# lstm parameters
input_size = 20
hidden_size = 64
num_layers = 1
# conv parameters
in_channels1 = 1
out_channels1 = 20
in_channels2 = 1
out_channels2 = 20
kernel_size1 = [1, 20]
stride1 = [1, 20]
kernel_size2 = [4, 4]
stride2 = [4, 4]
# linear parameters
in_features = 228
out_features1 = 100
out_features2 = 2





def main(train_model=True):

    data_dir = './data/'+ peptide_function + '/ACP240/'
    train_x = torch.load(data_dir + 'train_x.pt')
    train_y = torch.load(data_dir + 'train_y.pt')
    test_x = torch.load(data_dir + 'test_x.pt')
    test_y = torch.load(data_dir + 'test_y.pt')

    sequence_length = train_x.shape[1]
    if train_model:
        min_loss = 100000000000000000000000000000000.0
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        num_split = 1


        for train_index, val_index in skf.split(train_x, train_y):
            print('=============================================================')
            train_dataset = TensorDataset(train_x[train_index], train_y[train_index])
            val_dataset = TensorDataset(train_x[val_index], train_y[val_index])
            train_loader = DataLoader(dataset=train_dataset, batch_size=train_dataset.__len__(), shuffle=True, drop_last=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=val_dataset.__len__(), shuffle=True, drop_last=True)

            train_loss_list = []
            val_loss_list = []

            model = BiLSTM_CNN(sequence_length,
                               input_size, hidden_size, num_layers,
                               in_channels1, out_channels1, in_channels2, out_channels2, kernel_size1, stride1,
                               kernel_size2, stride2,
                               in_features, out_features1, out_features2).to(device)

            criterion = loss_for_sequence.CrossEntropy().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # scheduler = CosineAnnealingLR(optimizer, T_max=40, verbose=True)

            for epoch in range(num_epoch):
                epoch_train_loss = train(model, criterion, optimizer, train_loader, epoch, num_epoch)
                epoch_val_loss = eval(model, criterion, optimizer, val_loader, epoch, num_epoch)
                train_loss_list.append(epoch_train_loss)
                val_loss_list.append(epoch_val_loss)

                if epoch_val_loss < min_loss:
                    min_loss = epoch_val_loss
                    best_model = model.state_dict()
                    best_epoch = epoch
            model_dir = './tmp/model' + str(num_split) + '/'
            os.makedirs(model_dir)
            torch.save(best_model, os.path.join(model_dir, 'valid.pkl'))
            f = open('./tmp/model'+ str(num_split) + '/result.txt', 'a+')
            f.write('cross validation split' + str(num_split) + '\n')
            f.write('epoch:' + str(best_epoch+1) + '\n')
            f.write('loss=' + str(min_loss))
            f.close()

            plt.clf()
            plt.plot(train_loss_list, color="r", label="train loss")
            plt.plot(val_loss_list, color="b", label="val loss")
            plt.legend(loc="best")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig('./tmp/model{}/loss.png' .format(num_split))

            num_split = num_split + 1
    else:
        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(dataset=test_dataset, batch_size=test_dataset.__len__(), shuffle=True)

        model = BiLSTM_CNN(sequence_length,
                           input_size, hidden_size, num_layers,
                           in_channels1, out_channels1, in_channels2, out_channels2, kernel_size1, stride1,
                           kernel_size2, stride2,
                           in_features, out_features1, out_features2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_model_num = 5
        pretrain_model_dir = './tmp/model' + str(best_model_num) + '/valid.pkl'
        pretrain_model = torch.load(pretrain_model_dir)
        model.load_state_dict(pretrain_model)
        test(model, optimizer, test_loader)

main(train_model=True)
