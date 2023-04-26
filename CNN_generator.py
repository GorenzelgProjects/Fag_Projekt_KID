
from Network_calculation import *
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
#import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math


class Encoder1(nn.Module):
    
    def __init__(self,encoder_list, linear_int, in_channels, out_channels, encoded_space_dim,fc2_input_dim, layers=3, filter_size =32, kernel = (1,4), kernel_p = 2, stride = 2, stride_p = 2, padding = 1, padding_p = 0, pooling = False ):
        super().__init__()
        
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential()
            
        for i in range(layers):
            conv_name = str("Conv_{}".format(i+1))
            tanh_name = str("Tanh_{}".format(i+1))
            self.encoder_cnn.add_module(conv_name, nn.Conv2d(in_channels = in_channels[i], out_channels = out_channels[i], kernel_size=kernel, stride=stride, padding=padding)) #(1,251)
            self.encoder_cnn.add_module(tanh_name,nn.Tanh())
            if pooling:
                if i+1 < layers:
                    maxpool_name = str("Maxpool_{}".format(i+1))
                    self.encoder_cnn.add_module(maxpool_name, nn.MaxPool2d(kernel_size=kernel_p, padding=padding_p))
            
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(linear_int, 256),
            nn.Tanh(),
            nn.Linear(256, encoded_space_dim))
        
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class Decoder2(nn.Module):
    
    def __init__(self,decoder_list, linear_int, in_channels, out_channels, encoded_space_dim,fc2_input_dim, layers=3, filter_size =32, kernel = (1,4), kernel_p = 2, stride = 2, stride_p = 2, padding = 1, padding_p = 0, pooling = False ):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.Tanh(),
            nn.Linear(256, linear_int),
            nn.Tanh()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(out_channels[-1], 1, decoder_list[0]))

        self.decoder_conv = nn.Sequential()

        for i in range(layers):
            
            out_id = i * 2    
            conv_name = str("Conv_out_{}".format(i+1))
            tanh_name = str("Tanh_out_{}".format(i+1))
            self.decoder_conv.add_module(conv_name, nn.ConvTranspose2d(in_channels=in_channels[i], out_channels=out_channels[i], kernel_size=kernel, stride=stride, padding=padding, output_padding=0)) #(1,251)
            self.decoder_conv.add_module(tanh_name,nn.Tanh())
            if pooling:
                if i+1 < layers:
                    maxpool_name = str("Maxpool_upsample_{}".format(i+1))
                    self.decoder_conv.add_module(maxpool_name, nn.Upsample(size=(1,decoder_list[out_id]), mode='bilinear'))
            
        
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

in_channels = [35,70,140]
out_channels = [70,140,280]

NW = Network_cal()
encoder_list,decoder_list,linear_int=NW.Calc_convelution(layers=3, filter_size = out_channels[-1], kernel = 4, kernel_p = 2, stride = 2, stride_p = 2, padding = 1, padding_p = 0, pooling = False)


E = Encoder1(encoder_list, linear_int,in_channels = in_channels, out_channels = out_channels, encoded_space_dim = 35, fc2_input_dim = 35, layers=3, filter_size =32, kernel = (1,4), kernel_p = (1,2), stride = 2, stride_p = 2, padding = (0,1), padding_p = (0,0), pooling = True)
D = Decoder2(decoder_list, linear_int,in_channels = in_channels, out_channels = out_channels, encoded_space_dim = 35, fc2_input_dim = 35, layers=3, filter_size =32, kernel = (1,4), kernel_p = (1,2), stride = 2, stride_p = 2, padding = (0,1), padding_p = (0,0), pooling = True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(E.to(device))
print(D.to(device))