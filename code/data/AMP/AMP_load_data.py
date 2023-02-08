import numpy as np
from sklearn.model_selection import train_test_split
import torch

np.set_printoptions(threshold=np.inf)

label = []
seq_data = []
seq_length_data = []
protein_index = 0
with open('AMP.txt', 'r') as fp:
    for line in fp:
        if line[0] == '>':
            label.append(1)
        else:
            seq = line[:-1]
            seq_length = len(seq)
            seq_data.append(seq)
            seq_length_data.append(seq_length)
            protein_index += 1
with open('non-AMP.txt', 'r') as fp:
    for line in fp:
        if line[0] == '>':
            label.append(0)
        else:
            seq = line[:-1]
            seq_length = len(seq)
            seq_data.append(seq)
            seq_length_data.append(seq_length)
            protein_index += 1

max_length = max(seq_length_data)
data = np.zeros(shape=(protein_index,max_length,20))
seq_index = 0
for sequence in seq_data:
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M','N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in sequence]
    onehot_encoded = list()
    length = len(sequence)
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    data[seq_index,:length,:] = onehot_encoded
    seq_index += 1

train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.1, random_state=0)

train_x = torch.tensor(train_x, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.int64)
test_y = torch.tensor(test_y, dtype=torch.int64)

torch.save(train_x, 'AMP_train_x.pt')
torch.save(test_x, 'AMP_test_x.pt')
torch.save(train_y, 'AMP_train_y.pt')
torch.save(test_y, 'AMP_test_y.pt')

















