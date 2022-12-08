import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, z_dim, channel_dim):
        """
        Deep Convolutional Generator based on the framework described in https://arxiv.org/pdf/1511.06434.pdf
        
        inputs
        - z_dim: length of random noise which is the input
        - channel_dim: how many channels to output
        returns:
        - out: tensor generated images (batch_size, 3, 32,32)
        """
        super(Generator, self).__init__()
        # Parameters
        self.z_dim = z_dim
        self.channel_dim = channel_dim
        
        # Modules
        self.layer_0 = nn.Sequential(nn.ConvTranspose2d(z_dim, 1024, kernel_size = 4, 
                                                        stride =1, padding = 0, bias = False),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU())
        
        self.layer_1 = nn.Sequential(nn.ConvTranspose2d(in_channels = 1024, 
                                          out_channels = 512,
                                          kernel_size = 4, stride=2, 
                                          padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        
        self.layer_2 = nn.Sequential(nn.ConvTranspose2d(in_channels = 512, 
                                          out_channels = 256,
                                          kernel_size = 4, stride=2, 
                                          padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        
        self.layer_3 = nn.Sequential(nn.ConvTranspose2d(in_channels = 256, 
                                          out_channels = 128,
                                          kernel_size = 4, stride=2, 
                                          padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        
        self.layer_4 = nn.Sequential(nn.ConvTranspose2d(in_channels = 128, 
                                          out_channels = 64,
                                          kernel_size = 4, stride=1, 
                                          padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        
        self.layer_5 = nn.Sequential(nn.ConvTranspose2d(in_channels = 64, 
                                          out_channels = channel_dim,
                                          kernel_size = 2, stride=1, 
                                          padding=1, bias=False))
        
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        """
        runs the generator
        input is the random noise
        """
        l0 = self.layer_0(input)

        l1 = self.layer_1(l0)
        l2 = self.layer_2(l1)
        l3 = self.layer_3(l2)
        l4 = self.layer_4(l3)
        l5 = self.layer_5(l4)
        out = self.tanh(l5)
        return out