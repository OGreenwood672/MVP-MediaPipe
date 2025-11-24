import torch.nn as nn
import torch
import math


"""U-Net diffuser class

Encoding and decoding blocks
"""

class Encoder(nn.Module):
    def __init__(self,input_channels, output_channels, time_dimension = None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        if time_dimension is not None:
            self.time_layer = nn.Linear(time_dimension, self.output_channels)
        else:
            self.time_layer = None

        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(self.input_channels,self.output_channels,3,1,padding=1)
        self.conv2 = nn.Conv2d(self.output_channels,self.output_channels,3,1,padding=1)

    def forward(self,x,t=None):
        x = self.conv1(x)
        x = self.activation(x)
        if t is not None and self.time_layer:
            t = self.time_layer(t)
            t = self.activation(t)
            t = t.unsqueeze(-1).unsqueeze(-1)
            x = x + t
        x = self.conv2(x)
        x = self.activation(x)
        return x

class Decoder(nn.Module):
    def __init__(self,input_channels, output_channels, time_dimension=None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        if time_dimension:
            self.time_layer = nn.Linear(time_dimension, self.output_channels)
        else:
            self.time_layer = None

        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(self.input_channels,self.output_channels,3,1,padding=1)
        self.conv2 = nn.Conv2d(self.output_channels,self.output_channels,3,1,padding=1)

    def forward(self,x,t =None):
        x = self.conv1(x)
        x = self.activation(x)

        if t is not None and self.time_layer:
            t = self.time_layer(t)
            t = self.activation(t)
            t = t.unsqueeze(-1).unsqueeze(-1)
            x = x + t

        x = self.conv2(x)
        x = self.activation(x)
        return x

"""Time Embeddings"""

class TimeEmbeddings(nn.Module):
    def __init__(self,time_dimension):
        super().__init__()
        self.time_dimension = time_dimension

    def forward(self,t):
        half_dim = self.time_dimension//2
        freq_exponents = torch.arange(half_dim, device=t.device)
        frequencies = torch.exp(-math.log(10000) * freq_exponents / (half_dim - 1))
        t_frequencies = t[:, None] * frequencies[None, :]
        emb = torch.cat([t_frequencies.sin(), t_frequencies.cos()], dim=-1)
        return emb

"""Main U-Net defininition"""

class Unet(nn.Module):
    def __init__(self,input_channels, output_channels, num_joints, time_dimension=None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_dimension = time_dimension

        if time_dimension is not None:
            self.time_layer = nn.Sequential(TimeEmbeddings(time_dimension),nn.Linear(time_dimension, time_dimension),nn.ReLU())
            self.joint_embedding = nn.Embedding(num_joints, time_dimension)

        downs = []
        downs.append(Encoder(input_channels,64,time_dimension))
        downs.append(Encoder(64,128,time_dimension))
        downs.append(Encoder(128,256,time_dimension))
        downs.append(Encoder(256,512,time_dimension))
        downs.append(Encoder(512,1024,time_dimension))

        self.downs = nn.ModuleList(downs)
        self.pool = nn.MaxPool2d(2,2)


        ups = []
        ups.append(nn.ConvTranspose2d(1024,512,2,2))
        ups.append(Decoder(1024,512,time_dimension))
        ups.append(nn.ConvTranspose2d(512,256,2,2))
        ups.append(Decoder(512,256,time_dimension))
        ups.append(nn.ConvTranspose2d(256,128,2,2))
        ups.append(Decoder(256,128,time_dimension))
        ups.append(nn.ConvTranspose2d(128,64,2,2))
        ups.append(Decoder(128,64,time_dimension))

        self.ups = nn.ModuleList(ups)

        self.finalconv = nn.Conv2d(64,output_channels, 1)

    def forward(self, x, timestep=None, joint_index=None):

        t_emb = None
        if timestep is not None and self.time_dimension is not None:
            t_emb = self.time_layer(timestep)

        if joint_index is not None and hasattr(self, 'joint_embedding'):
            j_emb = self.joint_embedding(joint_index) # This was self.joint_layer before, which was incorrect
            if t_emb is not None:
                t_emb = t_emb + j_emb
            else:
                t_emb = j_emb

        residuals = []
        for down in self.downs[:-1]:
            x = down(x, t_emb)
            residuals.append(x)
            x = self.pool(x)
        x = self.downs[-1](x, t_emb)
        for up in self.ups:
            if isinstance(up,Decoder):
                res = residuals.pop()
                x = up(torch.cat((x,res),dim = 1), t_emb)
            else:
                x = up(x)
        x = self.finalconv(x)
        return x
