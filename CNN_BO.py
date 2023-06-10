from CNN import Epoch, Encoder, Decoder, dataload, calc_convolution
#from Save_BO import SaveData
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import GPyOpt
from CNN import Epoch, Encoder, Decoder, dataload, calc_convolution

encoded_space_dim = tuple([35,35])
layers = tuple([3,3])
kernel = tuple([4,4])
kernel_p = tuple([2,2]) 
stride = tuple([1,1]) 
stride_p = tuple([2,2]) 
padding = tuple([0,0])
padding_p = tuple([0,0]) 


#encoded_space_dim = tuple([35,35])                      #Ranges
#layers = tuple([3,3]) 
#kernel = tuple([2,4,6]) 
#kernel_p = tuple([2,4])  
#stride = tuple([1,2,3,4])  
#stride_p = tuple([1,2,3,4]) 
#padding = tuple([0,1,2]) 
#padding_p = tuple([0,0]) 

domain = [{'name': 'encoded_space_dim', 'type': 'discrete', 'domain': encoded_space_dim},
          {'name': 'layers', 'type': 'discrete', 'domain': layers},
          {'name': 'kernel', 'type': 'discrete', 'domain': kernel},
          {'name': 'kernel_p', 'type': 'discrete', 'domain': kernel_p},
          {'name': 'stride', 'type': 'discrete', 'domain': stride},
          {'name': 'stride_p', 'type': 'discrete', 'domain': stride_p},
          {'name': 'padding', 'type': 'discrete', 'domain': padding},
          {'name': 'padding_p', 'type': 'discrete', 'domain': padding_p}
         ]

train_loader, valid_loader, test_loader,test_dataset,test_labels = dataload()

def objective_function(x):
    in_channels_e = [35,64,128]
    out_channels_e = [64,128,128]
    #print(x)
    # we have to handle the categorical variables that is convert 0/1 to labels
    # log2/sqrt and gini/entropy
    param = x[0]

    #print(param)
    #print(int(param[0]),param[1])
    #if param[11] == True:
        #pooling = True
    #else:
        #pooling = False
    # we have to handle the categorical variables
    #if param[2] == 0:
    #    max_f = 'log2'
    #elif param[2] == 1:
    # max_f = 'sqrt'
    #else:
    #  max_f = None

    #if param[3] == 0:
    #  crit = 'gini'
    #else:
    #crit = 'entropy'

    #create the model
    #in_channels_e = [int(param[0])]

    encoder_list, decoder_list, linear_int = calc_convolution(layers=int(param[1]), 
                                                              filter_size = int(out_channels_e[-1]), 
                                                              kernel = int(param[2]), 
                                                              kernel_p = int(param[3]), 
                                                              stride = int(param[4]), 
                                                              stride_p = int(param[5]), 
                                                              padding = int(param[6]), 
                                                              padding_p = int(param[7]), 
                                                              pooling = True)


    #out_channels_e = [int(encoder_list[0])]

    in_channels_d = []
    out_channels_d = []

    #print(encoder_list)
    #print(decoder_list)

    #for i in range(1,len(encoder_list)-1,2):
        #in_channels_e.append(int(encoder_list[i]))
        #out_channels_e.append(int(encoder_list[i+1]))

    in_channels_d = out_channels_e.copy()
    out_channels_d = in_channels_e.copy()

    in_channels_d.reverse()
    out_channels_d.reverse()
    
    #print(in_channels_e, out_channels_e)
    #print(in_channels_d, out_channels_d)
    
    encoder = Encoder(encoder_list, 
                linear_int,
                in_channels = in_channels_e, 
                out_channels = out_channels_e, 
                encoded_space_dim = int(param[0]), 
                layers=int(param[1]), 
                kernel = (1,int(param[2])), 
                kernel_p = (1,int(param[3])), 
                stride = int(param[4]), 
                padding = (0,int(param[6])), 
                padding_p = (0,int(param[7])), 
                pooling = True)
    
    decoder = Decoder(decoder_list, 
                linear_int,
                in_channels = in_channels_d, 
                out_channels = out_channels_d, 
                encoded_space_dim = int(param[0]), 
                layers=int(param[1]), 
                kernel = (1,int(param[2])), 
                stride = int(param[4]), 
                padding = (0,int(param[6])), 
                pooling = True)

    params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
    ]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # use cuda or cpu
    loss_fn = torch.nn.MSELoss()
    lr= 0.01
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    encoder.to(device)
    decoder.to(device)

    model = Epoch(encoder, decoder, device, train_loader, test_dataset, test_labels, loss_fn, optimizer, n=10)
    
    #model.to(device)
    model.train(num_epochs=4) #dataloader, loss_fn, optimizer,n=10))
    
    # fit the model 
    
    test_loss = model.diz_loss['val_loss']
    test_loss = test_loss[-1]
    #model.fit(Xtrain, ytrain)
    print("TEST",test_loss)
    return test_loss


opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                            domain = domain,         # box-constrains of the problem
                                            acquisition_type = 'EI',      # Select acquisition function MPI, EI, LCB
                                            )
opt.acquisition.exploration_weight=0.05

opt.run_optimization(max_iter = 2) 

x_best = opt.X[np.argmin(opt.Y)]
print(x_best)
#print("The best parameters obtained: batch_size=" + str(x_best[0]) + ", learning_rate=" + str(x_best[1]))
#return x_best, architecture, hyperparameters