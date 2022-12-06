import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, channels, dim):
        
        """
        runs the discriminator model as discribed in https://arxiv.org/pdf/1511.06434.pdf
        inputs:
        - channels: int of channels in the images
        - dim: x/y axis length of the image
        """
        super(Discriminator,self).__init__()
        #parameters
        
        #modules
        self.layer_0 = nn.Sequential(nn.Conv2d(channels, 
                                               dim, 
                                               kernel_size = 2, 
                                               stride= 1, 
                                               padding = 1, bias = False),
                                     nn.LeakyReLU(.2))  #(b, 3, 32, 32)
        
        self.layer_1 = nn.Sequential(nn.Conv2d(dim, dim*2, kernel_size = 4, stride= 1, padding = 1, bias = False),
                                     nn.BatchNorm2d(dim*2),
                                     nn.LeakyReLU(0.2))#64
        
        self.layer_2 = nn.Sequential(nn.Conv2d(dim*2, dim*2**2, kernel_size = 4, stride= 2, padding = 1, bias = False),
                                     nn.BatchNorm2d(dim*2**2),
                                     nn.LeakyReLU(0.2))#128
        self.layer_3 = nn.Sequential(nn.Conv2d(dim*2**2, dim*2**3, kernel_size = 4, stride= 2, padding = 1, bias = False),
                                     nn.BatchNorm2d(dim*2**3),
                                     nn.LeakyReLU(0.2))#256
        self.layer_4 = nn.Sequential(nn.Conv2d(dim*2**3, dim*2**4, kernel_size = 4, stride= 2, padding = 1, bias = False),
                                     nn.BatchNorm2d(dim*2**4),
                                     nn.LeakyReLU(0.2))#512
        self.layer_5 = nn.Sequential(nn.Conv2d(dim*2**4, dim*2**5, kernel_size = 4, stride= 2, padding = 1, bias = False),
                                     nn.BatchNorm2d(dim*2**5),
                                     nn.LeakyReLU(0.2))#1024
        
        self.fully_connected = nn.Sequential(nn.Conv2d(dim*2**5, 1, kernel_size = 4, stride= 1, padding = 1, bias = False),
                                            nn.Sigmoid())
    def forward(self, input):
        l0 = self.layer_0(input)
        l1 = self.layer_1(l0)
        l2 = self.layer_2(l1)
        l3 = self.layer_3(l2)
        l4 = self.layer_4(l3)
        l5 = self.layer_5(l4)
        
        out = self.fully_connected(l5)
        return out[:,0,0,0]