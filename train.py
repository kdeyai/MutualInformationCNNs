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
import torch.utils.data as data_utils
import kde
import pickle

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
    for i in input:
        activity.append(torch.flatten(model.layer_activations[i][index],start_dim = 1, end_dim = -1))  
    return torch.cat(activity)    #need to concat batches
    


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    indices = torch.arange(20000)
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    index_sub = np.random.choice(np.arange(len(dataset1)), 20000, replace=False)
    dataset1.data = dataset1.data[index_sub]
    dataset1.targets = torch.tensor(dataset1.targets)[index_sub]

    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    saved_labelixs = {}
    for i in range(10):
        saved_labelixs[i] = np.squeeze(torch.tensor(dataset1.targets) == i)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if not os.path.isdir("models"):
        os.mkdir("models")
    
    CORRECT = 0
    nats2bits = 1.0/np.log(2) 
    scheduler = StepLR(optimizer, step_size=2, gamma=args.gamma)
    # MI_client = MI(X_train_subset, y_train_subset, 10)
    # MI_client.discretize()
    # MI_client.pre_compute()
    measures = OrderedDict()
    activation = 'relu'
    measures[activation] = {}
    y = F.one_hot(dataset1.targets, num_classes=10).detach().cpu().numpy()
    
    labelprobs = np.mean(y, axis=0)
    noise_variance = 1e-1 

   
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        cepochdata = defaultdict(list)
        PLOT_LAYERS =[] 
        input_act = []
        for i in model.input_data:
            input_act.append(torch.flatten(i,start_dim =1, end_dim =-1))
 
        in_act = torch.cat(input_act)
        for i in range(model.len):
            PLOT_LAYERS.append(i)   #this will plot for all layers
            activity = findactivity(model,i)
            h_upper = kde.entropy_estimator_kl(activity, noise_variance)
            h_lower = kde.entropy_estimator_bd(activity, noise_variance)
            hM_given_X = kde.kde_condentropy(activity, noise_variance)
            hM_given_Y_upper=0
            for i in range(10):
                hcond_upper = kde.entropy_estimator_kl(activity[saved_labelixs[i],:], noise_variance)
                hM_given_Y_upper += labelprobs[i] * hcond_upper
                
            hM_given_Y_lower=0.
            for i in range(10):
                hcond_lower = kde.entropy_estimator_bd(activity[saved_labelixs[i],:], noise_variance)
                hM_given_Y_lower += labelprobs[i] * hcond_lower

            binHM,binXM,binX_M, binYM,binY_M = simplebin.bin_calc_information2(in_act,saved_labelixs, activity, 0.67)   #calculating mutual information
            cepochdata['MI_XM_bin'].append( nats2bits * binXM )
            cepochdata['MI_YM_bin'].append( nats2bits * binYM )
            cepochdata['H_M_bin'].append(nats2bits * binHM)

            cepochdata['MI_XM_upper'].append( nats2bits * (h_upper - hM_given_X) )
            cepochdata['MI_YM_upper'].append( nats2bits * (h_upper - hM_given_Y_upper) )
            cepochdata['H_M_upper'  ].append( nats2bits * h_upper )

            cepochdata['MI_XM_lower'].append( nats2bits * (h_lower - hM_given_X) )
            cepochdata['MI_YM_lower'].append( nats2bits * (h_lower - hM_given_Y_lower) )
            cepochdata['H_M_lower'  ].append( nats2bits * h_lower )

        correct = test(model, device, test_loader)
        scheduler.step()
        measures[activation][epoch] = cepochdata
        model.input_data.clear()
        model.layer_activations.clear()
        # MI_client.mi_single_epoch(hidden_layers, epoch)
        
        torch.save(model.state_dict(), "models/mnist_cnn_%d.pt" % epoch)
        if correct > CORRECT:
            torch.save(model.state_dict(), "models/mnist_cnn_best.pt")
    ip.plotinformationplane(measures,PLOT_LAYERS)
    # with open('objs.pkl', 'w') as f:  
    #       pickle.dump([measures,PLOT_LAYERS], f)

if __name__ == '__main__':
    main()
