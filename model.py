import torch
from torch import nn 
from torch.nn import functional as F
from collections import defaultdict


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

        self.layer_activations = defaultdict(list)
        self.input_data = []
        self.len = 0

    def forward(self, x):
        y = x
        self.input_data.append(x)
        x = self.conv1(x)
        x = F.relu(x)

        self.layer_activations[y].append(x)

        x = self.conv2(x)
        x = F.relu(x)
 
        self.layer_activations[y].append(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
 
        self.layer_activations[y].append(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
 
        self.layer_activations[y].append(x)

        # x = self.dropout2(x)
        x = self.fc2(x)
 
        self.layer_activations[y].append(x)

        output = F.log_softmax(x, dim=1)
 
        self.layer_activations[y].append(output)

        self.len = len(self.layer_activations[y])
        return output

class MutualInfo(Net):
    def __init__(self):
        super().__init__()
        
        self.layer_map = ["conv", "conv", "dropout", "fc", "dropout", "fc", "ypred"]
        
        x = torch.ones((1,1,28,28))
        self.forward(x)

        print ([x.shape for x in self.layer_activations])

if __name__ == "__main__":

    mi = MutualInfo()

