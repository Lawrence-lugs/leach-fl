from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub
import torch

class ConvBNRelu( nn.Sequential ):
    """
    Convolutional layer with batchnorm and clipping ReLU
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,padding='same'):
        #padding = (kernel_size - 1) // 2
        super(ConvBNRelu, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, groups=groups, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

class DS_block( nn.Sequential ):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        super(DS_block, self).__init__(
            ConvBNRelu(in_planes,out_planes,kernel_size=kernel_size,groups=out_planes,stride=stride),
            ConvBNRelu(out_planes,out_planes,kernel_size=1)
        )

class DS_residual( nn.Module ):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        super(DS_residual, self).__init__()
        self.block = DS_block(in_planes, out_planes, kernel_size, stride)

    def forward(self, x):
        x = x + self.block(x)
        return x

class DS_CNN(nn.Module):
    '''
    Pytorch implementation of keyword 
    NAS <- wtf complicated models

    C(64,10,4,2,2)-DSC(64,3,1)-
    DSC(64,3,1)-DSC(64,3,1)-
    DSC(64,3,1)-AvgPool

    + A classifier- kek.
    '''
    def __init__(self):
        super(DS_CNN,self).__init__()
        self.encoder = nn.Sequential(
            ConvBNRelu(1,64,(10,4),(2,2),padding=0),
            nn.Dropout(0.2),
            DS_block(64,64,3,1),
            DS_block(64,64,3,1),
            DS_block(64,64,3,1),
            DS_block(64,64,3,1),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Linear(64,12)

    def forward(self,x):
        x = self.encoder(x)
        x = x.mean(-1).mean(-1)
        x = self.classifier(x)
        return x
        