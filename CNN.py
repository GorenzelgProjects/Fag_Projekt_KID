#from Network_calculation import *
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
#import pandas as pd 
#import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math

'''CNN main module with encoder class, decoder class, training function and EEG-dataloader. All of these 
classes are used in other modules, such as Bayesian Optimization, "CNN_BO.py", and is meant for a better overview.
If run in main module, it will run one training of the CNN-AE, with the settings manually set in the bottom of the file.'''


class Encoder(nn.Module):
    
    def __init__(self,encoder_list, linear_int, in_channels, out_channels, encoded_space_dim, layers=3, kernel = (1,4), kernel_p = 2, stride = 2, padding = 1, padding_p = 0, pooling = False):
        super().__init__()
        
        
                                                                        # Convolutional section
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
            
        
                                                                        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
                                                                        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(linear_int, 256),
            nn.Tanh(),
            nn.Linear(256, encoded_space_dim))
        
        
    def forward(self, x):
        #print(x.shape[:])
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class Decoder(nn.Module):
    
    def __init__(self,decoder_list, linear_int, in_channels, out_channels, encoded_space_dim, layers=3, kernel = (1,4), stride = 2, padding = 1, pooling = False ):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.Tanh(),
            nn.Linear(256, linear_int),
            nn.Tanh()
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(out_channels[0], 1, decoder_list[0]))

        self.decoder_conv = nn.Sequential()

        for i in range(layers):
            
            out_id = i * 2 +1
            conv_name = str("Conv_out_{}".format(i+1))
            tanh_name = str("Tanh_out_{}".format(i+1))
            self.decoder_conv.add_module(conv_name, nn.ConvTranspose2d(in_channels=in_channels[i], out_channels=out_channels[i], kernel_size=kernel, stride=stride, padding=padding, output_padding=0)) #(1,251)
            self.decoder_conv.add_module(tanh_name,nn.Tanh())
            if pooling:
                if i+1 < layers:
                    maxpool_name = str("Maxpool_upsample_{}".format(i+1))
                    self.decoder_conv.add_module(maxpool_name, nn.Upsample(size=(1,decoder_list[out_id]), mode='bilinear'))
                else:
                    maxpool_name = str("Maxpool_upsample_{}".format(i+1))
                    self.decoder_conv.add_module(maxpool_name, nn.Upsample(scale_factor=2, mode='bilinear'))
            
        
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

class Epoch:
    
    def __init__(self, encoder, decoder, device, dataloader,test_dataset,test_labels, loss_fn, optimizer,n=10):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dataloader = dataloader
        self.test_dataset = test_dataset
        self.test_labels = test_labels
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n = n
        super().__init__()

                                                                        ### Training function
    def train_epoch(self,verbose=False):
        
        self.encoder.train()                                            # Set train mode for both the encoder and the decoder
        self.decoder.train()
        train_loss = []
                                                                        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, _ in self.dataloader:                          # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            
            image_batch = image_batch.swapaxes(1,2)
            #image_batch = image_batch.swapaxes(2,3)
            image_batch = image_batch.to(self.device)
            
            #print(image_batch.shape[:])

            #image_batch = image_batch.swapaxes(1,2)
            #image_batch = image_batch.swapaxes(2,3)
            #print(image_batch.shape[:])
            
            
            encoded_data = self.encoder(image_batch)                    # Encode data
            
            decoded_data = self.decoder(encoded_data)                   # Decode data
            
            loss = self.loss_fn(decoded_data, image_batch)              # Evaluate loss
                                                                
            self.optimizer.zero_grad()                                  # Backward pass
            loss.backward()
            self.optimizer.step()
           
            if verbose == True:                                         # Print batch loss if "verbose" is set to True
                print('\t partial train loss (single batch): %f' % (loss.data))
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test_epoch(self):
        
        self.encoder.eval()                                             # Set evaluation mode for encoder
        self.decoder.eval()                                             # Set evaluation mode for decoder
        with torch.no_grad():                                           # Not tracking the gradients
            
            conc_out = []                                               # Define the lists to store the outputs for each batch
            conc_label = []
            for image_batch, _ in self.dataloader:
                
                image_batch = image_batch.swapaxes(1,2)                 # Move tensor to the proper device
                #image_batch = image_batch.swapaxes(2,3)
                image_batch = image_batch.to(self.device)               # Move tensor to the proper device

                #print(image_batch.shape[:])
                
                encoded_data = self.encoder(image_batch)                # Encode data
                
                decoded_data = self.decoder(encoded_data)               # Decode data
                
                conc_out.append(decoded_data.cpu())                     # Append the network output and the original image to the lists
                conc_label.append(image_batch.cpu())
            
            conc_out = torch.cat(conc_out)                              # Create a single tensor with all the values in the lists
            conc_label = torch.cat(conc_label) 
            
            val_loss = self.loss_fn(conc_out, conc_label)               # Evaluate global loss
        return val_loss.data
    
    def train(self,num_epochs=2,verbose=False):                         # Train function
        self.diz_loss = {'train_loss':[],'val_loss':[]}
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(verbose=verbose)
            val_loss = self.test_epoch()
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
            self.diz_loss['train_loss'].append(train_loss)              # Appending train loss 
            self.diz_loss['val_loss'].append(val_loss)                  # Appending validation loss 
            #self.plot_ae_outputs()
        
    
    def plot_ae_outputs(self):                                          #Main plot function
        plt.figure(figsize=(16,4.5))
        #targets = test_dataset.targets.numpy()
        targets = self.test_labels[:,0]
        #print(test_dataset.targets)
        t_idx = {i:np.where(targets==i)[0][0] for i in range(1,self.n)}
        #print(t_idx)
        for i in range(1,self.n):
            ax = plt.subplot(2,self.n,i+1)
            #img = test_data[t_idx[i]][0].unsqueeze(0).to(device)
            img = self.test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
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
        #test_transform = transforms.Compose([
        #transforms.ToTensor(),
        #])
        
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def calc_convolution(input_size=256, layers=3, filter_size =32, kernel = 4, kernel_p = 2, stride = 2, stride_p = 2, padding = 1, padding_p = 0, pooling = True):
        
        encoder = []
        for i,layer in enumerate(range(layers)):
            output_size = (input_size+2*padding - kernel)//stride + 1
            #output_size = ((input_size+2*(padding-1)*(kernel-1))-1//stride) + 1
            encoder.append(output_size)
            if pooling:
                if i+1 < layers:
                    output_size = (output_size+2*padding_p - kernel_p)//stride_p + 1
                    #output_size = ((output_size+2*(padding_p-1)*(kernel_p-1))-1//stride_p) + 1
                    encoder.append(output_size)
                
            
            input_size = output_size
        
        decoder = encoder.copy()
        decoder.reverse()
        
        linear = encoder[-1] * filter_size
        
        return encoder, decoder, linear

def dataload():

    train_data = np.load("train_data_1_30.npy")
    train_labels = np.load("train_label_1_30.npy").astype(np.int32)
    test_data = np.load("test_data_31_40.npy")
    test_labels = np.load("test_label_31_40.npy").astype(np.int32)

    batch_size=32

    #train_data = np.transpose(train_data, (1,2,0))

    train_data = train_data*1e5
    test_data = test_data*1e5
    train_dataset = EEGDataset(train_data, train_labels)
    test_dataset = EEGDataset(test_data, test_labels)


    train_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    m=len(train_dataset)

    #print(test_dataset[:,1])

    train_data, val_data = random_split(train_dataset, [math.floor(m*0.8), math.ceil(m*0.2)])
    #print(train_data[:5])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    return train_loader, valid_loader, test_loader, test_dataset, test_labels

if __name__ == "__main__":
    #in_channels = [35,70,140]
    #out_channels = [70,140,280]

    train_loader, valid_loader, test_loader,test_dataset,test_labels = dataload()

    in_channels_e = [35,64,128]
    out_channels_e = [64,128,128]
    encoder_list, decoder_list, linear_int = calc_convolution(layers=3, filter_size = out_channels_e[-1], kernel = 4, kernel_p = 2, stride = 2, stride_p = 2, padding = 1, padding_p = 0, pooling = True)


    #print(encoder_list, decoder_list)
    #out_channels_e = [encoder_list[0]]

    in_channels_d = []
    out_channels_d = []

    #for i in range(1,len(encoder_list)-1,2):
        #in_channels_e.append(encoder_list[i])
        #out_channels_e.append(encoder_list[i+1])

    in_channels_d = out_channels_e.copy()
    out_channels_d = in_channels_e.copy()

    in_channels_d.reverse()
    out_channels_d.reverse()
    
    #print(in_channels_e, out_channels_e)
    #print(in_channels_d, out_channels_d)

    encoder = Encoder(encoder_list, linear_int,in_channels = in_channels_e, out_channels = out_channels_e, encoded_space_dim = 35, layers=3, kernel = (1,4), kernel_p = (1,2), stride = 2, padding = (0,1), padding_p = (0,0), pooling = True)
    decoder = Decoder(decoder_list, linear_int,in_channels = in_channels_d, out_channels = out_channels_d, encoded_space_dim = 35, layers=3, kernel = (1,4), stride = 2, padding = (0,1), pooling = True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print(E.to(device))
    #print(D.to(device))

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
    model2 = model.train(num_epochs=4) #dataloader, loss_fn, optimizer,n=10))

    #print(encoder_list)
    #print(decoder_list)