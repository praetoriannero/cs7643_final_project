from AnoGAN.model.Generator import Generator
from AnoGAN.model.Discriminator import Discriminator
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchmetrics.functional.classification import binary_auroc, binary_accuracy

class AnoGAN(nn.Module):
    def __init__(self, z_dim = 100, channel_dim = 3, image_dim =32, lr = 0.002, beta1 = 0.5):
        super(AnoGAN, self).__init__()
        self.z = z_dim
        self.channel = channel_dim
        self.image_dim = image_dim
        self.lr = lr
        self.beta = beta1
        self.loss = nn.BCELoss()
        
        self.G = Generator(z_dim = z_dim, channel_dim = 3)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr = self.lr, betas=(self.beta, 0.999))
        self.D = Discriminator(channels = 3, dim = 32)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr = self.lr, betas=(self.beta, 0.999))
        
        
    def train(self, inputs, verbose = True):
        batch, y = inputs
        image = Variable(batch)
        batch_size = int(batch.size()[0])

        #Discriminator
        ones = Variable(torch.ones(batch_size))
        zeros = Variable(torch.zeros(batch_size))
        self.D_optimizer.zero_grad()
        z = Variable(torch.randn(batch_size, self.z, 1, 1))
        G_image = self.G.forward(z)
        D_fake = self.D.forward(G_image)
        D_fake_loss = self.loss(D_fake, zeros)
        D_real = self.D.forward(image)
        D_real_loss = self.loss(D_real, y)

        D_loss = D_fake_loss + D_real_loss 
        D_real_loss.backward()
        D_fake_loss.backward()
        
        self.D_optimizer.step()
        
        #Generator
        self.G_optimizer.zero_grad()
        z = Variable(torch.randn(batch_size, self.z, 1, 1))
        G_Fake = self.D.forward(self.G.forward(z))
        G_loss = self.loss(G_Fake, ones)
        G_loss.backward(retain_graph=True)
        self.G_optimizer.step()
        
        if verbose:
            return binary_auroc(D_real, y), binary_accuracy(D_real, y), float(G_loss.data), float(D_loss.data), float(D_fake_loss.data), float(D_real_loss.data)
        
        
    def predict(self, inputs, verbose = True):
        batch, y = inputs
        image = Variable(batch)
        batch_size = int(batch.size()[0])

        #Discriminator
        ones = Variable(torch.ones(batch_size))
        zeros = Variable(torch.zeros(batch_size))
        self.D_optimizer.zero_grad()
        z = Variable(torch.randn(batch_size, self.z, 1, 1))
        G_image = self.G.forward(z)
        D_fake = self.D.forward(G_image)
        D_fake_loss = self.loss(D_fake, zeros)
        D_real = self.D.forward(image)
        D_real_loss = self.loss(D_real, y)

        D_loss = D_fake_loss + D_real_loss 
        
        #Generator
        self.G_optimizer.zero_grad()
        z = Variable(torch.randn(batch_size, self.z, 1, 1))
        G_Fake = self.D.forward(self.G.forward(z))
        G_loss = self.loss(G_Fake, ones)
        
        if verbose:
            return D_real, binary_auroc(D_real, y), binary_accuracy(D_real[y ==1], y[y ==1]), float(G_loss.data), float(D_loss.data), float(D_fake_loss.data), float(D_real_loss.data)
        else:
            return D_real