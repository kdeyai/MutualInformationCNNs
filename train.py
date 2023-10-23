from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import simplebin
from collections import defaultdict, OrderedDict
import informationplane as ip


import os
from model import Net

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct

def findactivity(model, index):

    input = model.input_data
    activity = []
    print(len(input), input[0].size())
    for i in input:
        print(model.layer_activations[i][index].size())
        activity.append(model.layer_activations[i][index].flatten())  
    return torch.cat(activity)    #need to concat batches
    


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    saved_labelixs = {}
    for i in range(10):
        saved_labelixs[i] = np.squeeze(dataset1.targets == i)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    if not os.path.isdir("models"):
        os.mkdir("models")
    
    CORRECT = 0
    nats2bits = 1.0/np.log(2) 
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # MI_client = MI(X_train_subset, y_train_subset, 10)
    # MI_client.discretize()
    # MI_client.pre_compute()
    measures = OrderedDict()
    activation = 'relu'
    measures[activation] = {}
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        cepochdata = defaultdict(list)
        PLOT_LAYERS =[] 
        for i in range(model.len):
            PLOT_LAYERS.append(i)   #this will plot for all layers
            activity = findactivity(model,i)
            binHM,binXM,binX_M, binYM,binY_M = simplebin.bin_calc_information2(model.input_data,saved_labelixs, activity, 0.67)   #calculating mutual information
            cepochdata['MI_XM_bin'].append( nats2bits * binXM )
            cepochdata['MI_YM_bin'].append( nats2bits * binYM )
            cepochdata['H_M_bin'].append(nats2bits * binHM)
        correct = test(model, device, test_loader)
        scheduler.step()
        measures[activation][epoch] = cepochdata
        ip.plotinformationplane(measures,PLOT_LAYERS)
        # MI_client.mi_single_epoch(hidden_layers, epoch)
        
        torch.save(model.state_dict(), "models/mnist_cnn_%d.pt" % epoch)
        if correct > CORRECT:
            torch.save(model.state_dict(), "models/mnist_cnn_best.pt")

if __name__ == '__main__':
    main()
