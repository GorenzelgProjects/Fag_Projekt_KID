import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split, Dataset
import numpy as np
from torchvision import transforms
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import os
sns.set()
import time
from tqdm import tqdm


'''This file contains our main models, main datahandling and main plotting. This file is both used by itself to do single trainings
and transfer training and to be exported to other file such as "AE_1D_BO_T t.py" to do the BO training loops. In the bottom of the
file, there's a "if name == main" part, only runs when running directly from this file.'''


'''Main Transformer enchanced CAE model. Set up as a class for easier call. The input in the initiator is pretty straight forward.'''

class AET(nn.Module):
    def __init__(self,encoder_list=0,decoder_list=0, transformer_in=0, in_channels=[35,64], out_channels=[64,128], nhead=8, layers=3, kernel = 3, kernel_p = 2, stride = 1, stride_p = 2,padding = 1, padding_p = 0, pooling = True):
        super(AET, self).__init__()

        #in_channels = [35,64]
        #out_channels= [64,128]
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel, stride=stride, padding=padding),
            #nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=kernel_p, stride=stride_p),
            #nn.AvgPool1d(kernel_size=kernel_p, stride=stride_p),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel, stride=stride, padding=padding),
            #nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=kernel_p, stride=stride_p),
            #nn.AvgPool1d(kernel_size=kernel_p, stride=stride_p),
            #nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel, stride=stride, padding=padding),
            #nn.ReLU(),
            #nn.Tanh(),
        )

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=transformer_in, nhead=nhead, dim_feedforward=out_channels[-1], dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoderLayer(d_model=transformer_in, nhead=nhead, dim_feedforward=out_channels[-1], dropout=0.1)
        
        #self.transformer_e = nn.Sequential(
         #   nn.TransformerEncoder(self.transformer_encoder, num_layers=3))
        
        #self.transformer_d = nn.Sequential(
          #  nn.TransformerDecoder(self.transformer_decoder, num_layers=3))
        
        self.transformer_e = nn.TransformerEncoder(self.transformer_encoder, num_layers=3)
        
        self.transformer_d = nn.TransformerDecoder(self.transformer_decoder, num_layers=3)
        
        
        
        # Decoder layers
        self.decoder = nn.Sequential(
            #nn.Upsample(size=decoder_list[0], mode='nearest'),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Conv1d(in_channels=out_channels[2], out_channels=in_channels[2], kernel_size=kernel, stride=stride, padding=padding),

            
            nn.Upsample(size=decoder_list[0], mode='nearest'),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels[1], out_channels=in_channels[1], kernel_size=kernel, stride=stride, padding=padding),
            
            
            nn.Upsample(size=decoder_list[1], mode='nearest'),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels[0], out_channels=in_channels[0], kernel_size=kernel, stride=stride, padding=padding)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        transform_e = self.transformer_e(encoded)
        #transform_e = self.transformer_encoder
        transform_d = self.transformer_d(transform_e,encoded)
        decoded = self.decoder(transform_d)
        return decoded

'''Main CAE model. Set up as a class for easier call. The input in the initiator is pretty straight forward and is very alike
the above model (TECAE).'''


class AE(nn.Module):
    def __init__(self,encoder_list=0,decoder_list=0, transformer_in=0, in_channels=[35,64], out_channels=[64,128], nhead=0, layers=3, kernel = 3, kernel_p = 2, stride = 1, stride_p = 2,padding = 1, padding_p = 0, pooling = True):
        super(AE, self).__init__()

        #in_channels = [35,64]
        #out_channels= [64,128]
        
        # Encoder layers
        self.encoder = nn.Sequential(
            #nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel, stride=stride, padding=padding),
            #nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=kernel_p, stride=stride_p),
            #nn.AvgPool1d(kernel_size=kernel_p, stride=stride_p),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel, stride=stride, padding=padding),
            #nn.ReLU(),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=kernel_p, stride=stride_p),
            #nn.AvgPool1d(kernel_size=kernel_p, stride=stride_p),
            #nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel, stride=stride, padding=padding),
            #nn.ReLU(),
            #nn.Tanh(),
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            #nn.Upsample(size=decoder_list[0], mode='nearest'),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Conv1d(in_channels=out_channels[2], out_channels=in_channels[2], kernel_size=kernel, stride=stride, padding=padding),

            
            nn.Upsample(size=decoder_list[0], mode='nearest'),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels[1], out_channels=in_channels[1], kernel_size=kernel, stride=stride, padding=padding),
            
            
            nn.Upsample(size=decoder_list[1], mode='nearest'),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels[0], out_channels=in_channels[0], kernel_size=kernel, stride=stride, padding=padding)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


'''The main training class. Handles both the training and test over x-epochs, but also have data-loading-, data-sorting-
and plotting functions'''

class Epoch:
    
    def __init__(self, autoencoder, device, dataloader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, loss_fn, optimizer,test_dataset, test_labels,n=10, PATH='', paradigm=False):
    #def __init__(self, autoencoder, device, dataloader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, loss_fn, optimizer,n=10):
        self.model = autoencoder
        self.device = device
        self.dataloader = dataloader
        self.val_loader = valid_loader
        self.test_loader = test_loader
        self.avg_dataset = avg_dataset
        self.avg_dataset_test = avg_dataset_test
        self.avg = avg
        self.avg_test = avg_test
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.test_dataset = test_dataset
        self.test_labels = test_labels
        self.n = n
        self.PATH = PATH

        if paradigm:
            self.bin_diff = 11
            self.sub_diff = 1
            self.target_val = 11
        else:
            self.bin_diff = 1
            self.sub_diff = 31
            self.target_val = 1

        super().__init__()

    ### Training function
    def train_epoch(self,verbose=False):
        # Set train mode for both the encoder and the decoder
        train_loss = []
        val_loss = []
        for data in self.dataloader:
            inputs, label = data  # Assuming your dataloader provides (input, label) pairs
            #inputs = inputs.swapaxes(0,1)
            inputs = inputs.to(self.device)
            avg_outputs = self.avg_dataset[:][0]       
            avg_outputs = avg_outputs[label[:,1]-1,label[:,0]-1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs_device = outputs.to(self.device)
            loss = self.loss_fn(avg_outputs, outputs_device)  # Reconstruction loss
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())

        return np.mean(train_loss)

    def test_epoch(self):
        # Set evaluation mode for encoder and decoder
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            test_loss = []

            for test_data in self.test_loader:
                inputs, label = test_data  # Assuming your dataloader provides (input, label) pairs
                #inputs = inputs.swapaxes(0,1)
                inputs = inputs.to(self.device)

                avg_outputs_test = self.avg_dataset_test[:][0]
        
                avg_outputs_test = avg_outputs_test[label[:,1]-self.sub_diff,label[:,0]-self.bin_diff].to(self.device)

                outputs = self.model(inputs)

                outputs_device = outputs.to(self.device)

                loss = self.loss_fn(avg_outputs_test, outputs_device)  # Reconstruction loss
                test_loss.append(loss.item())
                
            return np.mean(test_loss)

    def test_epoch_2(self):
        # Set evaluation mode for encoder and decoder
        with torch.no_grad(): # No need to track the gradients
            # Define the lists to store the outputs for each batch
            test_loss = []
            paradigm_1 = []
            paradigm_2 = []
            paradigm_3 = []
            paradigm_4 = []
            paradigm_5 = []
            for test_data in self.test_loader:
                inputs, label = test_data  # Assuming your dataloader provides (input, label) pairs
                #inputs = inputs.swapaxes(0,1)
                inputs = inputs.to(self.device)

                avg_outputs_test = self.avg_dataset_test[:][0]
        
                avg_outputs_test = avg_outputs_test[label[:,1]-self.sub_diff,label[:,0]-self.bin_diff].to(self.device)

                outputs = self.model(inputs)

                outputs_device = outputs.to(self.device)

                for i, l in enumerate(label):
                    loss = self.loss_fn(avg_outputs_test[i], outputs_device[i])

                    if l[0] <= 4:
                        paradigm_1.append(loss.item())
                    elif l[0] == 5 or l[0] == 6:
                        paradigm_2.append(loss.item())
                    elif l[0] == 7 or l[0] == 8:
                        paradigm_3.append(loss.item())
                    elif l[0] == 9 or l[0] == 10:
                        paradigm_4.append(loss.item())
                    else:
                        paradigm_5.append(loss.item())

                #loss = self.loss_fn(avg_outputs_test, outputs_device)  # Reconstruction loss
                #test_loss.append(loss.item())

            return  [np.mean(paradigm_1), np.var(paradigm_1), np.std(paradigm_1)], [np.mean(paradigm_2), np.var(paradigm_2), np.std(paradigm_1)], [np.mean(paradigm_3), np.var(paradigm_3), np.std(paradigm_1)], [np.mean(paradigm_4), np.var(paradigm_4), np.std(paradigm_1)], [np.mean(paradigm_5), np.var(paradigm_5), np.std(paradigm_1)]


    
    def train(self,num_epochs=2,verbose=False):
        print("Initiating training...")
        self.diz_loss = {'train_loss':[],'val_loss':[]}
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(verbose=verbose)
            val_loss = self.test_epoch()
            print('\n EPOCH {}/{} \t train loss {} \t test loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
            self.diz_loss['train_loss'].append(train_loss)
            self.diz_loss['val_loss'].append(val_loss)

        #self.plot_ae_outputs()
        if verbose:
            self.path_plot = self.PATH + "/" + str(val_loss)
            os.makedirs(self.path_plot)
            self.plot_channels()
            self.plot_losses()

    
    def plot_ae_outputs(self):
        plt.figure(figsize=(16,4.5))
        #targets = test_dataset.targets.numpy()
        targets = self.test_labels[:,0]
        #print(targets)
        #print(test_dataset.targets)
        t_idx = {i:np.where(targets==i)[0] for i in range(1,self.n)}
        #print(t_idx)
        for i in range(1,self.n):
            ax = plt.subplot(2,self.n,i+1)
            ax.set_ylim(-4, 4)
            #img = test_data[t_idx[i]][0].unsqueeze(0).to(device)
                
            val = t_idx[i][0]

            img = self.test_dataset[val][0].unsqueeze(0)
            #img = img.swapaxes(1,2)

            avg_outputs_test = self.avg_dataset_test[:][0]
            
            avg_outputs_test = avg_outputs_test[0,i-1]

            with torch.no_grad():
                rec_img  = self.model(img.to(self.device))
            #plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.plot(avg_outputs_test)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(True)
                
            if i == self.n//2:
                ax.set_title('Average images')
            ax = plt.subplot(2, self.n, i + 1 + self.n)
            ax.set_ylim(-4, 4)
            #plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray') 
            #ax.plot(rec_img.squeeze(0).numpy())
            ax.plot(rec_img.cpu().squeeze().numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(True) 
                
            if i == self.n//2:
                ax.set_title('Reconstructed images')
        plt.show()
        
    def plot_channels(self):
        label_pos = []
        label_name = ["FP1","F3","F7","FC3","C3",
                       "C5","P3","P7","PO7","PO3",
                       "O1","Oz","Pz","CPz","FP2",
                       "Fz","F4","F8","FC4","FCz",
                       "Cz","C4","C6","P4","P8",
                       "PO8","PO4","O2","HEOG_left","HEOG_right",
                       "VEOG_lower","(corr)HEOG","(corr)VEOG_lower","(uncorr)HEOG","(uncorr)VEOG"]

        targets = self.test_labels[:,0]
        t_idx = {1:np.where(targets==self.target_val)[0]}
        val = t_idx[1][0]
        original = self.test_dataset[val][0].numpy()
        img = self.test_dataset[val][0].unsqueeze(0)

        avg_outputs_test = self.avg_dataset_test[:][0]
        avg_outputs_norm = avg_outputs_test[0,0].numpy()
        minimum = avg_outputs_norm.min(axis=1)
        maximum = avg_outputs_norm.max(axis=1)

        avg_outputs_test = avg_outputs_test[0,0].numpy()

        avg_outputs = self.avg_dataset[:][0]       
        avg_outputs = avg_outputs[0,0].numpy()

        with torch.no_grad():
            rec_img  = self.model(img.to(self.device))

        rec = rec_img.cpu().squeeze().numpy()

        for i in range(35):
            c = rec[i,:]
            v = avg_outputs_test[i,:]
            o = original[i,:]
            x = avg_outputs[i,:]
            rec[i,:] = (c - minimum[i]) / (maximum[i] - minimum[i])
            avg_outputs_test[i,:] = (v - minimum[i]) / (maximum[i] - minimum[i])
            original[i,:] = (o - minimum[i]) / (maximum[i] - minimum[i])
            avg_outputs[i,:] = (x - minimum[i]) / (maximum[i] - minimum[i])

        x = np.arange(0,256)
        for i in range(35):
            displacement = i*2

            plt.plot(x,original[i,:]+displacement, color='grey', alpha=0.5, linewidth=0.6)
            label_pos.append(avg_outputs_test[i,:].mean()+displacement)
            

        plt.yticks(label_pos, label_name, fontsize=9)
        plt.xlim([0, 256])
        temp_name = self.path_plot + "/" + "orginal.png"
        plt.savefig(temp_name, dpi=500)
        plt.clf()
        label_pos = []
        for i in range(35):
            displacement = i*2
            plt.plot(x,avg_outputs_test[i,:]+displacement, color='black',alpha=0.7, linewidth=0.6)
            label_pos.append(avg_outputs_test[i,:].mean()+displacement)

        plt.yticks(label_pos, label_name, fontsize=9)
        plt.xlim([0, 256])
        temp_name = self.path_plot + "/" + "average.png"
        plt.savefig(temp_name, dpi=500)
        plt.clf()

        label_pos = []
        for i in range(35):
            displacement = i*2
            plt.plot(x,avg_outputs_test[i,:]+displacement, color='black',alpha=0.7, linewidth=0.6)
            plt.plot(x,rec[i,:]+displacement, color='tomato', alpha=0.7, linewidth=0.6)
            label_pos.append(avg_outputs_test[i,:].mean()+displacement)

        plt.yticks(label_pos, label_name, fontsize=9)
        plt.xlim([0, 256])
        temp_name = self.path_plot + "/" + "avg_recon.png"
        plt.savefig(temp_name, dpi=500)
        plt.clf()

        label_pos = []
        for i in range(35):
            displacement = i*2

            plt.plot(x,rec[i,:]+displacement, color='tomato', alpha=0.7, linewidth=0.6)
            label_pos.append(avg_outputs_test[i,:].mean()+displacement)

        plt.yticks(label_pos, label_name, fontsize=9)
        plt.xlim([0, 256])
        temp_name = self.path_plot + "/" + "recon.png"
        plt.savefig(temp_name, dpi=500)
        plt.clf()

    def plot_losses(self):
        # Plot losses
        plt.figure(figsize=(10,8))
        plt.semilogy(self.diz_loss['train_loss'], label='Train')
        plt.semilogy(self.diz_loss['val_loss'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        #plt.grid()
        plt.legend()
        #plt.title('loss')
        temp_name = self.path_plot + "/" + "loss.png"
        plt.savefig(temp_name, dpi=500)
        plt.clf()


class EEGDataset(Dataset):
    def __init__(self, data, labels):
        #test_transform = transforms.Compose([
        #transforms.ToTensor(),
        #])
        #self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.data = torch.tensor(data, dtype=torch.float32).squeeze(1)
        #self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class EEGDataset2(Dataset):
    def __init__(self, data, labels):
        #test_transform = transforms.Compose([
        #transforms.ToTensor(),
        #])
        
        self.data = torch.tensor(data, dtype=torch.float32)
        #self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def average(data,labels,n):
    avg = np.zeros((30,n,35,256))
    avg_labels = np.zeros((30,2))
    for i in range(1,n+1):
        bins = labels[np.where(labels[:,0] == i)]
        bins_data = data[np.where(labels[:,0] == i)]
        for j in range(1,31):
            subs = bins[np.where(bins[:,1] == j)]
            subs_data = bins_data[np.where(bins[:,1] == j)]
            avg_labels[j-1,0] = i
            avg_labels[j-1,1] = j
            mean = subs_data.mean(axis=0)

            
            avg[j-1,i-1,:,:] = mean

    return avg, avg_labels

def average_2(data,labels,n):
    avg = np.zeros((10,n,35,256))
    avg_labels = np.zeros((10,2))
    for i in range(1,n+1):
        bins = labels[np.where(labels[:,0] == i)]
        bins_data = data[np.where(labels[:,0] == i)]
        for j in range(1,11):
            subs = bins[np.where(bins[:,1] == j+30)]
            subs_data = bins_data[np.where(bins[:,1] == j+30)]
            avg_labels[j-1,0] = i
            avg_labels[j-1,1] = j+30
            mean = subs_data.mean(axis=0)

            
            avg[j-1,i-1,:,:] = mean

    return avg, avg_labels

def paradigm_split(data,labels,n):
    
    print("Splitting data into paradigms")
    
    first_draw = True
    first_test = True
    
    avg = np.zeros((40,n-2,35,256))
    avg_labels = np.zeros((40,2))
    
    avg_test = np.zeros((40,2,35,256))
    avg_labels_test = np.zeros((40,2))
    
    #for i in range(1,n+1):
    for i in tqdm(range(1,n+1)):
        
            
        bins = labels[np.where(labels[:,0] == i)]
        bins_data = data[np.where(labels[:,0] == i)]

        #avg_labels[0,0] = i
        #avg_labels[0,1] = target_sub
        for j in range(1,41):
            subs = bins[np.where(bins[:,1] == j)]
            subs_data = bins_data[np.where(bins[:,1] == j)]
            #avg_labels[j-1,0] = i
            #avg_labels[j-1,1] = j
            #mean = subs_data.mean(axis=0)

            
            #avg[j-1,i-1,:,:] = mean
        
            if i > 10:
                if first_test:
                    test_data = subs_data
                    test_labels = subs
                    first_test = False
                else:
                    test_data = np.vstack((test_data, subs_data))
                    test_labels = np.vstack((test_labels, subs))
                    
                avg_test[j-1,i-11,:,:] = subs_data.mean(axis=0)
                
                #for j in range(1,41):
                avg_labels_test[j-1,0] = i
                avg_labels_test[j-1,1] = j
            else:
                if first_draw:
                    train_data = subs_data
                    train_labels = subs
                    first_draw = False
                else:
                    train_data = np.vstack((train_data, subs_data))
                    train_labels = np.vstack((train_labels, subs))
                    
                avg[j-1,i-1,:,:] = subs_data.mean(axis=0)
                
                #for j in range(1,41):
                avg_labels[j-1,0] = i
                avg_labels[j-1,1] = j

        #avg[0,i-1,:,:] = subs_data[1:transfer+1,:,:].mean(axis=0)


    return train_data, train_labels, test_data, test_labels, avg, avg_labels, avg_test, avg_labels_test

def reject_subjects(data,labels,reject_list,test=False, paradigm=False):

    first_draw = True

    if test:
        end = 11
        add = 30
    else:
        end = 31
        add = 0
    
    if paradigm:
        end = 41
        add = 0
    
    for i in tqdm(range(1,13)):
        bins = labels[np.where(labels[:,0] == i)]
        bins_data = data[np.where(labels[:,0] == i)]
        bin_idx = np.where(labels[:,0] == i)[0]
        for j in range(1,end):
            if j+add not in reject_list[i-1]:
                subs = bins[np.where(bins[:,1] == j+add)]
                subs_data = bins_data[np.where(bins[:,1] == j+add)]
                sub_idx = bin_idx[np.where(bins[:,1] == j+add)]

                if first_draw:
                    transfer_data = subs_data
                    transfer_labels = subs
                    sub_data = sub_idx
                    first_draw = False
            
                else:
                    transfer_data = np.vstack((transfer_data, subs_data))
                    transfer_labels = np.vstack((transfer_labels, subs))
                    sub_data = np.append(sub_data, sub_idx)

    return transfer_data, transfer_labels

def test_transfer(data,labels,n,transfer=0, target_bin=1, target_sub=31):
    first_draw = True
    avg = np.zeros((1,n,35,256))
    avg_labels = np.zeros((1,2))
    #for i in range(1,n+1):
    for i in range(1,n+1):
        bins = labels[np.where(labels[:,0] == i)]

        bin_idx = np.where(labels[:,0] == i)[0]
        bins_data = data[np.where(labels[:,0] == i)]
        subs = bins[np.where(bins[:,1] == target_sub)]

        sub_idx = bin_idx[np.where(bins[:,1] == target_sub)]

        subs_data = bins_data[np.where(bins[:,1] == target_sub)]
        avg_labels[0,0] = i
        avg_labels[0,1] = target_sub
        
        if len(subs_data) > 0:
            if transfer > subs_data.shape[0]:
                transfer = subs_data.shape[0]

            random_idx = np.random.choice(subs_data.shape[0], transfer, replace=False)

            if first_draw:
                transfer_data = subs_data[random_idx,:,:]
                transfer_labels = subs[random_idx,:]
                #sub_data = sub_idx[random_idx]
                first_draw = False
            else:
                transfer_data = np.vstack((transfer_data, subs_data[random_idx,:,:]))
                transfer_labels = np.vstack((transfer_labels, subs[random_idx,:]))
                #sub_data = np.append(sub_data, sub_idx[random_idx])

            avg[0,i-1,:,:] = subs_data[random_idx,:,:].mean(axis=0)


    return transfer_data, transfer_labels, avg, avg_labels, data, labels

def calc_convolution(input_size=256,layers=2, filter_size =32, kernel = 4, kernel_p = 2, stride = 2, stride_p = 2, padding = 1, padding_p = 0, pooling = False):
        
        encoder = []
        for i,layer in enumerate(range(layers)):
            output_size = (input_size+2*padding - kernel)//stride + 1
            encoder.append(output_size)
            if pooling:
                if i+1 < layers:
                    output_size = (output_size+2*padding_p - kernel_p)//stride_p + 1
                    encoder.append(output_size)
                else:
                    transformer_in = (output_size+2*padding_p - kernel_p)//stride_p + 1
            input_size = output_size
        
        decoder = encoder.copy()
        decoder.reverse()
        
        last_output = decoder[-1]
        for i in range(1,len(decoder)-1,2):
            decoder.pop(i)
            encoder.pop(i)

        decoder[-1] = 256 - (last_output - 256)
        #decoder[-1] -= last_output - 256
        linear = encoder[-1] * filter_size
        
        return encoder, decoder, transformer_in

def plot_mse(avg, rec, transfer_list):
    plt.plot(transfer_list, avg, label="average_loss")
    plt.scatter(transfer_list, avg)
    plt.plot(transfer_list, rec, label="reconstruction_loss")
    plt.scatter(transfer_list, rec)
    plt.legend()
    plt.show()


def dataload_init(n=12, paradigm=False, reject=False):
    print("Loading data from npy files")
    
    reject_list = np.array([[0,0,0,1,16,5],
                                [0,0,0,1,16,5],
                                [0,0,0,1,16,5],
                                [0,0,0,1,16,5],
                                [0,0,0,0,0,7],
                                [0,0,0,0,0,7],
                                [0,7,9,10,12,28],
                                [0,7,9,10,12,28],
                                [0,0,0,0,0,40],
                                [0,0,0,0,0,40],
                                [6,9,10,30,35,40],
                                [6,9,10,30,35,40]])
    
    if paradigm:
        all_data = np.load("all_data.npy")
        all_labels = np.load("all_labels.npy").astype(np.int32)

        train_data, train_labels, test_data, test_labels, avg, avg_labels, avg_test, avg_labels_test = paradigm_split(all_data, all_labels, n)
        
        train_data = train_data*1e5
        test_data = test_data*1e5
        print("Paradigm ",train_data.shape[:])
        
        if reject:
            print("Removing rejected subjects from train data")
            train_data, train_labels = reject_subjects(train_data,train_labels,reject_list,test=False,paradigm=True)
            print("Removing rejected subjects from test data")
            test_data, test_labels = reject_subjects(test_data,test_labels,reject_list,test=True,paradigm=True)
        
        print("Paradigm ",train_data.shape[:])

    else:
        train_data = np.load("train_data_1_30.npy")
        train_labels = np.load("train_label_1_30.npy").astype(np.int32)
        test_data = np.load("test_data_31_40.npy")
        test_labels = np.load("test_label_31_40.npy").astype(np.int32)
    
        train_data = train_data*1e5
        test_data = test_data*1e5

        avg, avg_labels = average(train_data, train_labels, n)

        avg_test, avg_labels_test = average_2(test_data, test_labels, n)

        

        if reject:
            print("Removing rejected subjects from train data")
            train_data, train_labels = reject_subjects(train_data,train_labels,reject_list,test=False,paradigm=False)
            print("Removing rejected subjects from test data")
            test_data, test_labels = reject_subjects(test_data,test_labels,reject_list,test=True,paradigm=False)

    return train_data, train_labels, test_data, test_labels, avg, avg_labels, avg_test, avg_labels_test


def dataload(train_data, train_labels, test_data, test_labels, avg, avg_labels, avg_test, avg_labels_test, batch_size = 32, n = 12, transfer = 0):

    if transfer > 0:
        for sub in tqdm(range(31,41,1)):
            transfer_data, transfer_labels, transfer_avg_data, transfer_avg_labels, test_data, test_labels = test_transfer(data=test_data, labels=test_labels, n=n, transfer=transfer, target_bin=1, target_sub=sub)
            #print(transfer_data.shape[:])
            train_data = np.vstack((train_data, transfer_data))
            train_labels = np.vstack((train_labels, transfer_labels))
            avg = np.vstack((avg, transfer_avg_data))
            avg_labels = np.vstack((avg_labels, transfer_avg_labels))
            test_data = test_data.copy()
            test_labels = test_labels.copy()

    train_dataset = EEGDataset(train_data, train_labels)
    test_dataset = EEGDataset(test_data, test_labels)
    #transfer_dataset = EEGDataset(transfer_data, transfer_labels)

    avg_dataset = EEGDataset2(avg, avg_labels)
    avg_dataset_test = EEGDataset2(avg_test, avg_labels_test)

    train_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transfer_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    #transfer_dataset.transform = transfer_transform

    m=len(train_dataset)

    #print(test_dataset[:,1])
    train_data = train_dataset
    val_data = train_dataset
    #train_data, val_data = random_split(train_dataset, [math.floor(m*0.8), math.ceil(m*0.2)])
    #print(train_data[:5])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    return train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, test_dataset, test_labels


'''This part is only run, when running this specific .py file. Used for single training runs, transfer training and paradigm-based training.'''

if __name__ == "__main__":

    path = "./AET_plots_til_martin"           #The plots will be saved at this relative path in a folder with the given name.

    paradigm = False            #Set to true if we want to train over paradigms
    reject = True               #Set to true if we want to rejects "bad" subject/datapoints

    in_channels = [35,128]      #fixed input channels
    out_channels= [128,128]     #fixed ouput channels

    layers=2
    kernel = 3
    kernel_p = 2
    stride = 1
    stride_p = 2
    padding = 1
    padding_p = 0
    pooling = True              #Set to true for maxpooling
    nhead = 16                  #Number of attention heads

    num_epochs = 10             #Number of epochs
    batch_size = 32             
    n = 12                      #Number of labels
    transfer = 0                #Number of test trials that needs to be transfer 

    reject_list = np.array([[0,0,0,1,16,5],         #List of rejected subjects in the format of the data.
                                [0,0,0,1,16,5],
                                [0,0,0,1,16,5],
                                [0,0,0,1,16,5],
                                [0,0,0,0,0,7],
                                [0,0,0,0,0,7],
                                [0,7,9,10,12,28],
                                [0,7,9,10,12,28],
                                [0,0,0,0,0,40],
                                [0,0,0,0,0,40],
                                [6,9,10,30,35,40],
                                [6,9,10,30,35,40]])
    
    encoder_list, decoder_list, transformer_in = calc_convolution(layers=layers, kernel = kernel, kernel_p = kernel_p, stride = stride, stride_p = stride_p, padding = padding, padding_p = padding_p, pooling = pooling) #Stride can't be change do to BO
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")         #Autoselecting cuda/cpu

    train_data, train_labels, test_data, test_labels, avg, avg_labels, avg_test, avg_labels_test = dataload_init(n=12, paradigm=paradigm, reject=reject)            #Data initiator
    
    print(train_data.shape)
    print(test_data.shape)

    mse_transfer = []
    mse_loss_list = []
    mse_avg_loss = []

    paradigm_1 = []
    paradigm_2 = []
    paradigm_3 = []
    paradigm_4 = []
    paradigm_5 = []


    avg_paradigm_1 = []
    avg_paradigm_2 = []
    avg_paradigm_3 = []
    avg_paradigm_4 = []
    avg_paradigm_5 = []    
    
    for t in range(0,5,5):          #Transfor training loop. set to (range(0,5,5)) to do not transfer and a single training.
        print("Transfering: {} trials from test data to train data ".format(t))
        transfer = t

                    #Below are different optimizer / loss setting. Change comments to choose the appropiate settings.

        #autoencoder = AET(encoder_list=encoder_list,decoder_list=decoder_list, transformer_in=transformer_in, in_channels=in_channels, out_channels=out_channels, nhead=nhead, layers=layers, kernel = kernel, kernel_p = kernel_p, stride = stride, stride_p = stride_p,padding = padding, padding_p = padding_p, pooling = pooling)
        autoencoder = AET(encoder_list=encoder_list,decoder_list=decoder_list, transformer_in=transformer_in, in_channels=in_channels, out_channels=out_channels, nhead=nhead, layers=layers, kernel = kernel, kernel_p = kernel_p, stride = stride, stride_p = stride_p,padding = padding, padding_p = padding_p, pooling = pooling)
        autoencoder.to(device)
        #criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        #criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        #optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.01, momentum=0.9)

        temp_avg_1, temp_avg_2, temp_avg_3, temp_avg_4, temp_avg_5 = [], [], [], [], []

        train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg_target, avg_test_target, test_dataset, target_labels = dataload(train_data, train_labels, test_data, test_labels, avg, avg_labels, avg_test, avg_labels_test, batch_size=batch_size, n=n, transfer=transfer)        #Trainloader functions 
        
        if not paradigm:
            true_avg = avg_dataset_test[:][0].to(device)
            transfer_avg = avg_dataset[30:][0].to(device)
            for i,s in enumerate(transfer_avg):
                for j,b in enumerate(s):
                    loss = criterion(true_avg[i,j], b)
                    if loss.item() != 0 and i+31 not in reject_list[j]:
                        if j+1 <= 4:
                            temp_avg_1.append(loss.item())
                        elif j+1 == 5 or j+1 == 6:
                            temp_avg_2.append(loss.item())
                        elif j+1 == 7 or j+1 == 8:
                            temp_avg_3.append(loss.item())
                        elif j+1 == 9 or j+1 == 10:
                            temp_avg_4.append(loss.item())
                        else:
                            temp_avg_5.append(loss.item())

            avg_paradigm_1.append([np.mean(temp_avg_1),np.var(temp_avg_1),np.std(temp_avg_1)])
            avg_paradigm_2.append([np.mean(temp_avg_2),np.var(temp_avg_2),np.std(temp_avg_2)])
            avg_paradigm_3.append([np.mean(temp_avg_3),np.var(temp_avg_3),np.std(temp_avg_3)])
            avg_paradigm_4.append([np.mean(temp_avg_4),np.var(temp_avg_4),np.std(temp_avg_4)])
            avg_paradigm_5.append([np.mean(temp_avg_5),np.var(temp_avg_5),np.std(temp_avg_5)])

        #print(avg_paradigm_1)

        model = Epoch(autoencoder, device, train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg_target, avg_test_target, criterion, optimizer, test_dataset, target_labels, n=n, PATH=path, paradigm=paradigm)       #Loading model with the given settings as inputs        
        model2 = model.train(num_epochs=num_epochs, verbose=True) #dataloader, loss_fn, optimizer,n=10))            #Train function. set verbose to True if we want plots for individual training loops
        if not paradigm:
            (p1, p2, p3, p4, p5) = model.test_epoch_2()
        
            paradigm_1.append(p1)
            paradigm_2.append(p2)
            paradigm_3.append(p3)
            paradigm_4.append(p4)
            paradigm_5.append(p5)

        if transfer > 0:
            
            mse_transfer.append(i)
            mse_loss_list.append(model.diz_loss['val_loss'][-1])

            true_avg = avg_dataset_test[:][0].to(device)
            transfer_avg = avg_dataset[30:][0].to(device)
            
            loss = criterion(true_avg, transfer_avg)  # Reconstruction loss
            mse_avg_loss.append(loss.item())


    print("p1 (mean,var,std): {}".format(paradigm_1))
    print("-"*30)
    print("p2 (mean,var,std): {}".format(paradigm_2))
    print("-"*30)
    print("p3 (mean,var,std): {}".format(paradigm_3))
    print("-"*30)
    print("p4 (mean,var,std): {}".format(paradigm_4))
    print("-"*30)
    print("p5 (mean,var,std): {}".format(paradigm_5))
    print("_"*30)

    print("avg1 (mean,var,std): {}".format(avg_paradigm_1))
    print("-"*30)
    print("avg2 (mean,var,std): {}".format(avg_paradigm_2))
    print("-"*30)
    print("avg3 (mean,var,std): {}".format(avg_paradigm_3))
    print("-"*30)
    print("avg4 (mean,var,std): {}".format(avg_paradigm_4))
    print("-"*30)
    print("avg5 (mean,var,std): {}".format(avg_paradigm_5))

    plot_mse(mse_avg_loss, mse_loss_list, mse_transfer)