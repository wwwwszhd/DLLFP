import torch
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTM_CNN(nn.Module):
    def __init__(self, sequence_length,
                 input_size, hidden_size, num_layers,
                 in_channels1, out_channels1, in_channels2, out_channels2, kernel_size1, stride1, kernel_size2, stride2,
                 in_features, out_features1, out_features2
                 ):
        super().__init__()
        self.sequence_length = sequence_length

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bias=True, bidirectional=True)

        self.in_channels1 = in_channels1
        self.out_channels1 = out_channels1
        self.in_channels2 = in_channels2
        self.out_channels2 = out_channels2
        self.kernel_size1 = kernel_size1
        self.stride1 = stride1
        self.kernel_size2 = kernel_size2
        self.stride2 = stride2

        matrix = torch.load('blosum_matrix.pt')
        pssm = torch.reshape(matrix, [20, 1, 1, 20])
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels1, out_channels=self.out_channels1, kernel_size=self.kernel_size1, stride=self.stride1,bias=False)
        self.conv1.weight.data = Parameter(pssm)

        self.conv2 = torch.nn.Conv2d(in_channels=self.in_channels2, out_channels=self.out_channels2, kernel_size=self.kernel_size2, stride=self.stride2, bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.in_features = in_features
        self.out_features1 = out_features1
        self.out_features2 = out_features2
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=self.out_features1, bias=True)
        self.fc2 = nn.Linear(in_features=self.out_features1, out_features=self.out_features2, bias=True)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1):
        batch_size = np.shape(input1)[0]
        lstm_input = torch.reshape(input1, shape=(batch_size, self.sequence_length, self.input_size))
        _, (final_state, cell_state) = self.lstm(lstm_input)
        final_state = self.dropout(final_state)
        final_state_fw = final_state[0, :, :]
        final_state_bw = final_state[1, :, :]
        lstm_output = torch.cat((final_state_fw, final_state_bw), dim=1)
        lstm_output = torch.reshape(lstm_output, shape=(-1, 128))

        conv_input = torch.reshape(input1, shape=(batch_size, 1, self.sequence_length, self.input_size))
        conv1_output = self.conv1(conv_input)
        conv1_output = torch.reshape(conv1_output, shape=(batch_size, self.sequence_length, 1, 20))

        input2 = torch.sum(input1, dim=(1,2), dtype=torch.int32)
        bandwidth = torch.floor(torch.div(input2, 4))
        width = torch.mul(bandwidth, 4).int()
        avblock_output = torch.zeros((batch_size, 4, 20))
        for i in range(batch_size):
            avblock_temp = torch.reshape(conv1_output[i][0:width[i]], [4, -1, 20])
            avblock = torch.reshape(torch.mean(avblock_temp, dim=1), [1, 4, 20])
            avblock_output[i, :, :] = avblock

        avblock_output = torch.reshape(avblock_output, shape=(batch_size, 1, 4, self.input_size)).to(device)
        conv2_output = self.conv2(avblock_output)
        conv2_output = self.relu(conv2_output)
        conv2_output = self.dropout(conv2_output)
        conv_pssm_output = torch.reshape(conv2_output, shape=(-1, 100))

        merge_feature = torch.cat((lstm_output, conv_pssm_output), dim=1)

        fc1_output = self.fc1(merge_feature)
        fc2_output = self.fc2(fc1_output)
        output = self.softmax(fc2_output)

        return output