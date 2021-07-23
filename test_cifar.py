  
import argparse
import random

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 ResNet Test')



parser.add_argument('--normalization', default='torch_bn', type=str,
                    help='normalization type [bn, in, bin,ln,gn,nn,torch_bn]')

parser.add_argument('--n', default=2, type=int, 
                    help='number of layers [1,2,3]')

parser.add_argument('--output_file', default='trained_models_cifar/part_1.1_predictions.txt', type=str, metavar='PATH',
                    help='file containing the prediction in the same order as in the input csv')

parser.add_argument('--model_file', default='trained_models_cifar/part_1.1.pth', type=str, metavar='PATH',
                    help='path to the trained model')

parser.add_argument('--test_data_file', default='../data/cifar_test_data.csv ', type=str, metavar='PATH',
                    help='path to a csv with each line representating an image')

args = parser.parse_args()


# Random seed
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)


def main():


    # read the data
    print('Reading data....')
    df_test = pd.read_csv(args.test_data_file)
    test_images = df_test.iloc[:,:]

    transform = transforms.Compose([
    transforms.ToTensor()])



    test_data = CIFAR10Dataset(test_images, transform)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



   

    n = args.n
    norm = args.normalization
          
    model = ResNet(n,norm)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading saved model....")
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_file,map_location = torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.model_file))
    
    model.to(device)
    y_pred = evaluate(model,test_loader,device)
    labels = []
    for pred in y_pred:
        labels.append(classes[pred])

    # open file
    with open(args.output_file, 'w+') as f:
          
        # write elements of list
        for items in labels:
            f.write('%s\n' %items)
          
        print("File written successfully")
      
      
    # close the file
    f.close()


    
# custom dataset
class CIFAR10Dataset(Dataset):
    def __init__(self, images, transforms=None):
        self.X = images
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(32, 32, 3)
        
        if self.transforms:
            data = self.transforms(data)
        return data


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
        self.rho = nn.Parameter(torch.ones(shape))
         

    def forward(self, X):
        
        mean_i = X.mean(dim=(2, 3), keepdim=True)
        var_i = ((X - mean_i)**2).mean(dim=(2, 3), keepdim=True)
        X_hat_i = (X - mean_i) / torch.sqrt(var_i + self.eps)
      
        mean_b = X.mean(dim=(0, 2, 3), keepdim=True)
        var_b = ((X - mean_b)**2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat_b = (X - mean_b) / torch.sqrt(var_b + self.eps)
    
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
    




def evaluate(model, data_loader, device):

    y_pred = np.array([], dtype=int)
    
    with torch.no_grad():
        for data in data_loader:
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            

            y_pred = np.concatenate((y_pred, predicted.cpu()))
    
    return y_pred


if __name__ == '__main__':
    main()
