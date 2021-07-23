  
import argparse
import random

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import numpy as np
import os
import shutil


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 ResNet Training')



parser.add_argument('--normalization', default='torch_bn', type=str,
                    help='normalization type {bn, in, bin,ln,gn,nn,torch_nn}')

parser.add_argument('--n', default=2, type=int, 
                    help='number of layers [1,2,3]')

parser.add_argument('--data_dir', default='../data/cifar-10-batches-py', type=str, metavar='PATH',
                    help='directory containing data')

parser.add_argument('--output_file', default='trained_models_cifar/part_1.1.pth', type=str, metavar='PATH',
                    help='path to the trained model')

args = parser.parse_args()


# Random seed
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)


def main():



    if not os.path.exists('trained_models_cifar'):
        os.makedirs('trained_models_cifar')
        print('Folder created....')



    print('Preparing dataset....')
    train_transform = transforms.Compose([ 
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.ToTensor()])


    try:
        dataset =  datasets.CIFAR10(root=args.data_dir, train=True,
                                        download=False, transform=train_transform)
    except RuntimeError:
        dataset =  datasets.CIFAR10(root=args.data_dir, train=True,
                                        download=True, transform=train_transform)
          
    train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                              shuffle=True, num_workers=2,
                                                 pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=128,
                                              shuffle=True, num_workers=2,
                                                 pin_memory=True)


    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        
    epochs = 100

    # OPTIMISER PARAMETERS
    lr = 0.1 
    momentum = 0.9
    weight_decay = 0.0001 

    milestones = [52, 71]
    # Divide learning rate by 10 at each milestone
    gamma = 0.1





    n = args.n
    norm = args.normalization
    model_path = args.output_file
    

    print("Creating model....")
          
    model = ResNet(n,norm)

    criterion = torch.nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    print("Training model....")
    train(model, epochs, train_loader, valid_loader, criterion, 
              optimizer, scheduler,model_path)
    print('Training finished....')







def evaluate(model, data_loader, device):
    
    y_true = np.array([], dtype=int)
    y_pred = np.array([], dtype=int)
    
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true = np.concatenate((y_true, labels.cpu()))
            y_pred = np.concatenate((y_pred, predicted.cpu()))
    
    error = np.sum(y_pred != y_true) / len(y_true)
    return error



def train(model, epochs, train_loader, valid_loader, criterion, 
          optimizer, scheduler,model_path):
    
    # Run on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    #results_df.set_index('epoch')
    print('Epoch \tBatch \tNLLLoss_Train')
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        model.train()
        running_loss  = 0.0
        best_valid_err = 1.0
        for i, data in enumerate(train_loader, 0):   # Do a batch iteration
            
            # get the inputs
            inputs, labels = data
          
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print average loss for last 50 mini-batches
            running_loss += loss.item()
            if i % 50 == 49:
                print('%d \t%d \t%.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        
        
        scheduler.step()
        
        # Record metrics
        model.eval()
        train_loss = loss.item()
        train_err = evaluate(model, train_loader, device)
        valid_err = evaluate(model, valid_loader, device)
        print(f'train_err: {train_err} vaild_err: {valid_err}')
        
        # Save best model
        if valid_err < best_valid_err:
            torch.save(model.state_dict(), model_path)
            best_test_err = valid_err

    print('Finished Training')
    model.eval()

#Various Normalization Schemes

#Batch_normalization
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
            
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  
    return Y, moving_mean.data, moving_var.data
   
class BatchNorm(nn.Module):
    
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9)
        return Y


#Layer_normalization
class LayerNorm(nn.Module):
    def __init__(self, num_features, num_dims,eps=1e-5):
        super().__init__()
        self.eps = eps
        if num_dims == 2:
            shape = (1,num_features)
        else:
            shape = (1,num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X):

        if len(X.shape) == 2:
            mean = X.mean(dim=1)
            var = ((X - mean)**2).mean(dim=0)
        else:           
            mean = X.mean(dim=(1, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(1, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + self.eps)
        
        Y = self.gamma * X_hat + self.beta
        
        return Y


#Instance_normalization
class InstanceNorm(nn.Module):
    def __init__(self, num_features, num_dims,eps=1e-5):
        super().__init__()
        self.eps = eps
        if num_dims == 2:
            shape = (1,num_features)
        else:
            shape = (1,num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X):

        if len(X.shape) == 2:
            mean = X.mean(dim=1)
            var = ((X - mean)**2).mean(dim=0)
        else:           
            mean = X.mean(dim=(2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + self.eps)
        
        Y = self.gamma * X_hat + self.beta  
        
        return Y


#Group_normalization
class GroupNorm(nn.Module):
    def __init__(self, num_features,G, num_dims,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.G =G
        if num_dims == 2:
            shape = (1,num_features)
        else:
            shape = (1,num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    
    def forward(self, X):
        N,C,H,W = X.size()
        G = self.G

        X = X.view(N,G,-1)
        mean = X.mean(-1, keepdim=True)
        var = ((X - mean)**2).mean(-1, keepdim=True)

        X_hat =  (X - mean) / torch.sqrt(var + self.eps)
        X_hat = X_hat.view(N,C,H,W)
        Y = self.gamma * X_hat + self.beta  
        
        return Y

#Batch_instance_normalization
class BatchInstanceNorm(nn.Module):
    def __init__(self, num_features,eps=1e-5):
        super().__init__()
        self.eps = eps
        shape = (1,num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.rho = nn.Parameter(torch.Tensor(shape))
         

    def forward(self, X):
        
        mean_i = X.mean(dim=(2, 3), keepdim=True)
        var_i = ((X - mean)**2).mean(dim=(2, 3), keepdim=True)
        X_hat_i = (X - mean) / torch.sqrt(var + self.eps)

        mean_b = X.mean(dim=(0, 2, 3), keepdim=True)
        var_b = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat_b = (X - mean) / torch.sqrt(var + self.eps)

        rho_limited = torch.sigmoid(self.rho)      
        Y = self.gamma * (rho_limited * X_hat_b + (1-rho_limited)*X_hat_i) + self.beta  
        
        return Y

#ResNet Model

#block for creating 'n' number of stacks
class block(nn.Module):
    def __init__(self, filters,norm, down_sampling=False):
        super().__init__()

        s = 2 if down_sampling else 1
        
        self.norm = norm
        self.conv1 = nn.Conv2d(int(filters/s), filters, kernel_size=3, stride=s, padding=1, bias=False)
        
        if self.norm != 'nn':
          if self.norm == 'torch_bn':
            self.bn1 = nn.BatchNorm2d(filters, track_running_stats=True)
          elif self.norm == 'bn':
            self.bn1 = BatchNorm(filters,num_dims = 4)
          elif self.norm == 'ln':
            self.bn1 = LayerNorm(filters,num_dims = 4)
          elif self.norm == 'gn':
            self.bn1 = GroupNorm(filters,G = 8,num_dims = 4)
          elif self.norm == 'in':
            self.bn1 = InstanceNorm(filters,num_dims = 4)
          elif self.norm == 'bin':
            self.bn1 = BatchInstanceNorm(filters)
        
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        if self.norm != 'nn':
          if self.norm == 'torch_bn':
            self.bn2 = nn.BatchNorm2d(filters, track_running_stats=True)
          elif self.norm == 'bn':
            self.bn2 = BatchNorm(filters,num_dims = 4)
          elif self.norm == 'ln':
            self.bn2 = LayerNorm(filters,num_dims = 4)
          elif self.norm == 'gn':
            self.bn2 = GroupNorm(filters,G = 8,num_dims = 4)
          elif self.norm == 'in':
            self.bn2 = InstanceNorm(filters,num_dims = 4)
          elif self.norm == 'bin':
            self.bn2 = BatchInstanceNorm(filters)
        self.relu2 = nn.ReLU()

        self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)


    def shortcut(self, out, x):

        if x.shape != out.shape:
           
            d = self.downsample(x)
            p = torch.mul(d, 0)
            return out + torch.cat((d, p), dim=1)
        else:
            return out + x        
    
    def forward(self, x):
        
        out = self.conv1(x)
        if self.norm != 'nn':
          out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        if self.norm != 'nn':
          out = self.bn2(out)
          
        out = self.shortcut(out, x)
        out = self.relu2(out)
        
        return out
    

#ResNet with 6n+2 layers
class ResNet(nn.Module):
    def __init__(self, n, norm):
        super().__init__()

        self.norm = norm
        
        
        self.convIn = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        if self.norm != 'nn':
          if self.norm == 'torch_bn':
            self.bnIn = nn.BatchNorm2d(16, track_running_stats=True)
          elif self.norm == 'bn':
            self.bnIn = BatchNorm(16,num_dims = 4)
          elif self.norm == 'ln':
            self.bnIn = LayerNorm(16,num_dims = 4)
          elif self.norm == 'gn':
            self.bnIn = GroupNorm(16,G = 8,num_dims = 4)
          elif self.norm == 'in':
            self.bnIn = InstanceNorm(16,num_dims = 4)
          elif self.norm == 'bin':
            self.bnIn = BatchInstanceNorm(16)
        self.relu   = nn.ReLU()
        
    
        self.stack1 = nn.ModuleList([block(16,norm = self.norm, down_sampling=False) for _ in range(n)])

   
        self.stack2a = block(32,norm = self.norm, down_sampling=True)
        self.stack2b = nn.ModuleList([block(32,norm = self.norm, down_sampling=False) for _ in range(n-1)])

 
        self.stack3a = block(64,norm = self.norm, down_sampling=True)
        self.stack3b = nn.ModuleList([block(64,norm = self.norm, down_sampling=False) for _ in range(n-1)])
   
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcOut   = nn.Linear(64, 10, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)  
        
        
    def forward(self, x):  

        
        out = self.convIn(x)
      
        if self.norm != 'nn':
          out = self.bnIn(out)
        out = self.relu(out)
        
        for l in self.stack1:
            out = l(out)
        
        out = self.stack2a(out)
        for l in self.stack2b: 
            out = l(out)
        
        out = self.stack3a(out)
        for l in self.stack3b: 
            out = l(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fcOut(out)
        return self.softmax(out)
    

if __name__ == '__main__':
    main()
