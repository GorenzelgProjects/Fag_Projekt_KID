#from autoencoder_1D import Epoch, EEGAutoencoder, EEGDataset,EEGDataset2, dataload, calc_convolution
from AE_1D_T import Epoch, AET,AE, EEGDataset,EEGDataset2, dataload, calc_convolution
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
import GPyOpt


kernel = 3
layers = 2
kernel_p = 2
stride = 1
stride_p = 2 
padding = 1
padding_p = 0 
#criterion = nn.L1Loss()

batch_size=32
n = 12                   # Number of labels
transfer = 0


#nhead = tuple([4,8,16,32])
learning_rate = (1*1e-5,1)
weight_decay = (0.01,1)


#encoded_space_dim = tuple([35,35])                      #Ranges
#layers = tuple([3,3]) 
#kernel = tuple([2,4,6]) 
#kernel_p = tuple([2,4])  
#stride = tuple([1,2,3,4])  
#stride_p = tuple([1,2,3,4]) 
#padding = tuple([0,1,2]) 
#padding_p = tuple([0,0]) 

#domain = [{'name': 'nhead', 'type': 'discrete', 'domain': nhead},                               #0         
#          {'name': 'learning_rate', 'type': 'continuous', 'domain': learning_rate},               #1
#          {'name': 'weight_decay', 'type': 'continuous', 'domain': weight_decay},                 #2
#         ]

domain = [{'name': 'learning_rate', 'type': 'continuous', 'domain': learning_rate},               #0
          {'name': 'weight_decay', 'type': 'continuous', 'domain': weight_decay},                 #1
         ]

train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, test_dataset, test_labels = dataload(batch_size=batch_size, n=n, transfer=transfer)
dataloader = train_loader

def objective_function(x):
    param = x[0]
    in_channels = [35,128]
    out_channels= [128,128]
    encoder_list, decoder_list, transformer_in = calc_convolution(filter_size=int(35),
                                                     layers=int(layers), 
                                                     kernel = int(kernel), 
                                                     kernel_p = int(kernel_p), 
                                                     stride = int(stride), 
                                                     stride_p = int(stride_p), 
                                                     padding = int(padding), 
                                                     padding_p = int(padding_p), 
                                                     pooling = True)
    #print(x)
    # we have to handle the categorical variables that is convert 0/1 to labels
    # log2/sqrt and gini/entropy
    

    #elif param[4] == 2 or 3:
    #    criterion = nn.MSELoss()
    #elif param[4] == 3:
    #    criterion = nn.CosineEmbeddingLoss()
    #elif param[4] == 4:
    #    criterion = nn.CrossEntropyLoss()
    

    autoencoder = AE(encoder_list=0,
                decoder_list=decoder_list, 
                transformer_in=transformer_in,
                in_channels = in_channels, 
                out_channels = out_channels, 
                nhead = 0,#int(param[0]), 
                layers=int(layers), 
                kernel = int(kernel), 
                kernel_p = int(kernel_p), 
                stride = int(stride), 
                padding = int(padding), 
                padding_p = int(padding_p), 
                pooling = True)
    
   
    params_to_optimize = [
    {'params': autoencoder.parameters()},
    ]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #autoencoder = EEGAutoencoder()                                  #We've merely set these to 
    criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(autoencoder.parameters(), lr=param[0],weight_decay=param[1])
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=param[0],momentum=param[1])

    autoencoder.to(device)
   
    num_epochs = 8
    
    

    model = Epoch(autoencoder, device, train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, criterion, optimizer, test_dataset, test_labels, n=n)      
    #model.to(device)
    model.train(num_epochs=num_epochs) #dataloader, loss_fn, optimizer,n=10))

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

opt.run_optimization(max_iter = 15) 

x_best = opt.X[np.argmin(opt.Y)]
print(x_best)

      
#print("The best parameters obtained:" 
#      + domain[0]["name"] + "=" + str(x_best[0]) + "," 
#      + domain[1]["name"] + "=" + str(x_best[1]) + "," 
#      + domain[2]["name"] + "=" + str(x_best[2]))

print("The best parameters obtained:" 
      + domain[0]["name"] + "=" + str(x_best[0]) + "," 
      + domain[1]["name"] + "=" + str(x_best[1]))

try:
    opt.plot_acquisition()
except:
    pass
#return x_best, architecture, hyperparameters