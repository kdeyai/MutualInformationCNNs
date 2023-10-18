import numpy as np

import torch
from torch import nn 
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.layer_activations = []

    def forward(self, x):
        
        self.layer_activations.append(x)        

        x = self.conv1(x)
        x = F.relu(x)

        self.layer_activations.append(x)

        x = self.conv2(x)
        x = F.relu(x)
 
        self.layer_activations.append(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
 
        self.layer_activations.append(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
 
        self.layer_activations.append(x)

        x = self.dropout2(x)
        x = self.fc2(x)
 
        self.layer_activations.append(x)

        output = F.log_softmax(x, dim=1)
 
        self.layer_activations.append(output)

        return output

class MutualInfo(Net):
    def __init__(self, n_bins=40):
        super().__init__()        
        
        self.n_bins = n_bins
        #self.layer_map = ["conv", "conv", "dropout", "fc", "dropout", "fc", "ypred"]

    def bin(self, x, c):
        # n_bins bins = n_bins + 1 ranges; precision = 6 decimal places for float32 and arange
        bin_size = (max(x) - min(x)) / float(self.n_bins + 1)
        bins = np.arange(min(x), max(x) + bin_size/2., bin_size) 
        bins = np.round(bins[1:], decimals=6)
        
        binned_data = {}
        probs = [0 for _ in range(self.n_bins)]

        bin_idx = 0        
        
        for unique_x, x_count in zip(x, c):
            if bins[bin_idx] >= unique_x:
                if bin_idx in binned_data:
                    binned_data[bin_idx].append([unique_x, x_count])
                else:
                    binned_data[bin_idx] = [[unique_x, x_count]]
            else:
                if not bin_idx in binned_data:
                    binned_data[bin_idx] = [[np.nan,0]]
                bin_idx += 1               
        
        for bin_idx in range(self.n_bins):
            probs[bin_idx] = np.sum(np.array(binned_data[bin_idx])[:,1])
        probs = np.array(probs) 
        probs /= float(probs.sum())
        
    def calculate_mi(self, x, f, y):

        x, f, y = x.cpu().numpy(), f.cpu().detach().numpy(), y.cpu().detach().numpy()
        
        unique_x, counts_x = np.unique(x, return_counts=True)        
        p_x = counts_x / counts_x.sum()
        unique_y, counts_y = np.unique(y, return_counts=True)        
        p_y = counts_y / counts_y.sum()
        unique_f, counts_f = np.unique(f, return_counts=True)        
        p_f = counts_f / counts_f.sum()
        
        #self.bin(unique_x, counts_x)
        self.bin(unique_f, counts_f)
        self.bin(unique_y, counts_y)
        exit()      

if __name__ == "__main__":

    mi = MutualInfo()
        
    x = torch.ones((1,1,28,28))
    mi.forward(x)
    
    mi.calculate_mi(mi.layer_activations[0], mi.layer_activations[1], mi.layer_activations[-1]) 

