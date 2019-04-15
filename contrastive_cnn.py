#Models for Contrastive Convolution FR ECCV2018

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class network_4layers(nn.Module):   
   def __init__(self, num_classes = 10000):
       super(network_4layers, self).__init__()
       self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 64, kernel_size = 3, stride = 1, padding = 0)
       self.conv2 = nn.Conv2d(in_channels = 64 , out_channels = 128, kernel_size = 3, stride = 1, padding = 0)
       self.conv3 = nn.Conv2d(in_channels = 128 , out_channels = 256, kernel_size = 3, stride = 1, padding = 0)
       self.conv4 = nn.Conv2d(in_channels = 256 , out_channels = 512, kernel_size = 3, stride = 1, padding = 0)
       self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)

   def forward(self, x):
       x = F.relu(self.conv1(x))
       x = self.maxpool(x)

       x = F.relu(self.conv2(x))
       x = self.maxpool(x)

       x = F.relu(self.conv3(x))
       x = self.maxpool(x)

       x = F.relu(self.conv4(x))
       x = self.maxpool(x)


       return x

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1,padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=padding, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        #print('Inside block i/p size',x.size())
        out = self.conv1(x)
        #print('Inside block conv1 size',out.size())
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #print('Inside block conv2 size',out.size())
        #out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        #print('Inside block o/p size',out.size())
        return out

# ResNet based 10 layer
class network_10layers(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(network_10layers, self).__init__()
        self.in_channels = 128 
        self.conv1 = conv3x3(1, 128, padding = 0)
        self.conv2 = conv3x3(128,128, padding = 0)
        self.conv3 = conv3x3(128,256, padding = 0)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block,128, 128, layers[0])
        self.layer2 = self.make_layer(block,256, 256, layers[1])#, 2)
        
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv4 = conv3x3(256, 512,padding = 0)
    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        '''
        downsample = None
        
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride))
        '''  
        layers = []
        layers.append(block(in_channels, out_channels, stride))#, downsample))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        #print('ip',x.size()) 
        out = self.conv1(x)
        #print('Conv1 op',out.size()) 
        out = self.maxpool(out)
        #print('conv1+maxpool op',out.size()) 
        out = self.relu(out)
        out = self.conv2(out)
        #print('Conv2 op',out.size()) 
        out = self.layer1(out)
        #print('layer1 op',out.size()) 
        out = self.maxpool(out)
        #print('layer1 + maxpool op',out.size()) 
        out = self.conv3(out)
        #print('Conv3 op',out.size()) 
        out = self.layer2(out)
        #print('layer2 op',out.size()) 
        out = self.maxpool(out)
        #print('layer2 op+ maxpool',out.size()) 
        out = self.conv4(out)
        #print('Conv4 op',out.size()) 
        out = self.maxpool(out)
        return out

# ResNet based 14 layer
class network_14layers(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(network_14layers, self).__init__()
        self.in_channels = 128
        self.conv1 = conv3x3(1, 128, padding = 0)
        self.conv2 = conv3x3(128,128, padding = 0)
        self.conv3 = conv3x3(128,256, padding = 0)
        self.conv4 = conv3x3(256, 512,padding = 0)

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block,128, 128, layers[0])
        self.layer2 = self.make_layer(block,256, 256, layers[1])#, 2)
        self.layer3 = self.make_layer(block,512,512, layers[2])#, 2)

        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #print('ip',x.size())
        out = self.conv1(x)
        #print('Conv1 op',out.size())
        out = self.maxpool(out)
        #print('conv1+maxpool op',out.size())
        out = self.relu(out)
        out = self.conv2(out)
        #print('Conv2 op',out.size())
        out = self.layer1(out)
        #print('layer1 op',out.size())
        out = self.maxpool(out)
        #print('layer1 + maxpool op',out.size())
        out = self.conv3(out)
        #print('Conv3 op',out.size())
        out = self.layer2(out)
        #print('layer2 op',out.size())
        out = self.maxpool(out)
        #print('layer2 op+ maxpool',out.size())
        out = self.conv4(out)
        #print('Conv4 op',out.size())
        out = self.layer3(out)
        #print('layer3 op',out.size())
        out = self.maxpool(out)
        return out


def Contrastive_4Layers(**kwargs):
    model = network_4layers(**kwargs)
    return model

def Contrastive_10Layers(**kwargs):
    model = network_10layers(ResidualBlock,[1,2],**kwargs)
    return model

def Contrastive_14Layers(**kwargs):
    model = network_14layers(ResidualBlock,[2,3,1],**kwargs)
    return model

def main():
    #model = Contrastive_4Layers(num_classes = 8631)
    
    model = Contrastive_14Layers()#.to(device)
    print(model)
    x = torch.rand(1,1,128,128)
    y = model(x)
    print(y.size()) #torch.Size([1, 512, 4, 4])
    
if __name__ == '__main__':
    main()
