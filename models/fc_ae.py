from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub
import torch

class FCBNReLU( nn.Sequential ):
    """
    FC layer with batchnorm and clipping ReLU
    """
    def __init__(self, in_size, out_size):
        super(FCBNReLU, self).__init__(
            nn.Linear(in_size,out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(inplace=True)
        )

class FC_AE(nn.Module):
    '''
    Pytorch implementation of the DCASE2020 Task 2 Baseline

    Inputs: mel spectrogram
    Outputs: Reconstructed mel spectrogram
    '''
    def __init__(self, input_size=640, mid_size=128, bottleneck=8):
        super(FC_AE,self).__init__()
        self.encoder = nn.Sequential(
            FCBNReLU(input_size,mid_size),
            FCBNReLU(mid_size,mid_size),
            FCBNReLU(mid_size,mid_size),
            FCBNReLU(mid_size,mid_size),
            FCBNReLU(mid_size,bottleneck),
        )

        self.decoder = nn.Sequential(
            FCBNReLU(bottleneck,mid_size),
            FCBNReLU(mid_size,mid_size),
            FCBNReLU(mid_size,mid_size),
            FCBNReLU(mid_size,mid_size),
            nn.Linear(mid_size,input_size),
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self,x):
        x = self.quant(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.dequant(x)
        return x
        
    def fuse_model(self):
        for m in self.modules():
            if type(m) == FCBNReLU:
                torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)