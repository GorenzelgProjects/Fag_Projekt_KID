
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


class Encoder(nn.Module):
    
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
class Decoder(nn.Module):
    
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


class Epoch:
    
    def __init__(self, encoder, decoder, device, dataloader, loss_fn, optimizer,n=10):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n = n
        super().__init__()

    ### Training function
    def train_epoch(self,verbose=False):
        # Set train mode for both the encoder and the decoder
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in self.dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            
            image_batch = image_batch.swapaxes(1,2)
            
            image_batch = image_batch.to(self.device)
            # Encode data
            encoded_data = self.encoder(image_batch)
            # Decode data
            decoded_data = self.decoder(encoded_data)
            # Evaluate loss
            loss = self.loss_fn(decoded_data, image_batch)
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Print batch loss
            if verbose == True:
                print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test_epoch(self):
        # Set evaluation mode for encoder and decoder
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            for image_batch, _ in self.dataloader:
                # Move tensor to the proper device
                image_batch = image_batch.swapaxes(1,2)
                image_batch = image_batch.to(self.device)
                # Encode data
                encoded_data = self.encoder(image_batch)
                # Decode data
                decoded_data = self.decoder(encoded_data)
                # Append the network output and the original image to the lists
                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label) 
            # Evaluate global loss
            val_loss = self.loss_fn(conc_out, conc_label)
        return val_loss.data
    
    def train(self,num_epochs=2,verbose=False):
        self.diz_loss = {'train_loss':[],'val_loss':[]}
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(verbose=verbose)
            val_loss = self.test_epoch()
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
            self.diz_loss['train_loss'].append(train_loss)
            self.diz_loss['val_loss'].append(val_loss)
            self.plot_ae_outputs()
        
    
    def plot_ae_outputs(self):
        plt.figure(figsize=(16,4.5))
        #targets = test_dataset.targets.numpy()
        targets = test_labels[:,0]
        #print(test_dataset.targets)
        t_idx = {i:np.where(targets==i)[0][0] for i in range(1,self.n)}
        #print(t_idx)
        for i in range(1,self.n):
            ax = plt.subplot(2,self.n,i+1)
            #img = test_data[t_idx[i]][0].unsqueeze(0).to(device)
            img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
            img = img.swapaxes(1,2)
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                rec_img  = self.decoder(self.encoder(img))
            #plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.plot(img.cpu().squeeze().numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(True)  
            if i == self.n//2:
                ax.set_title('Original images')
            ax = plt.subplot(2, self.n, i + 1 + self.n)
            #plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray') 
            ax.plot(rec_img.cpu().squeeze().numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(True)  
            if i == self.n//2:
                ax.set_title('Reconstructed images')
        plt.show()  

    def plot_losses(self):
        # Plot losses
        plt.figure(figsize=(10,8))
        plt.semilogy(self.diz_loss['train_loss'], label='Train')
        plt.semilogy(self.diz_loss['val_loss'], label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        #plt.grid()
        plt.legend()
        #plt.title('loss')
        plt.show()


class EEGDataset(Dataset):
    def __init__(self, data, labels):
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        ])
        
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]