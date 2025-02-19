from torch import nn
import torch

class ConvBNRelu( nn.Sequential ):
    """
    Convolutional layer with batchnorm and clipping ReLU
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNRelu, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

class ResBlock( nn.Module ):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ResBlock,self).__init__()
        self.stride = stride
        padding = (kernel_size - 1) // 2
        self.convbnrelu = ConvBNRelu(in_planes, out_planes, kernel_size, stride, groups)
        self.convbn = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.conv_for_stride = nn.Conv2d(in_planes,out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.relu0 = nn.ReLU(inplace=True)

    def forward(self,x):
        y = self.convbnrelu(x)
        y = self.convbn(y)        

        if self.stride != 1:
            x = self.conv_for_stride(x)
        
        x = y + x
        x = self.relu0(x)
        return x

# MLPerfTiny baseline implementation applies a 1e-4 regularization
class MLPerfTiny_ResNet_Baseline( nn.Module ):
    def __init__(self,num_classes):
        super(MLPerfTiny_ResNet_Baseline,self).__init__()
        
        self.input_layer = ConvBNRelu(3, 16, 3, 1)  

        self.features = nn.Sequential(
            ResBlock(16,16,3,1),
            ResBlock(16,32,3,2),
            ResBlock(32,64,3,2),
        )

        self.classifier = nn.Linear(64,num_classes)
    
    def forward(self,x):
        x = self.input_layer(x)
        x = self.features(x)
        x = x.mean(-1).mean(-1)
        x = self.classifier(x)
        return x