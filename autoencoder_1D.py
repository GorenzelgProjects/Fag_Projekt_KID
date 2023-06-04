import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split, Dataset
import numpy as np
from torchvision import transforms
import math
import matplotlib.pyplot as plt

class EEGAutoencoder(nn.Module):
    def __init__(self):
        super(EEGAutoencoder, self).__init__()

        in_channels = 35
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            #nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            #nn.Tanh(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            #nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, in_channels, kernel_size=3, stride=1, padding=1),
            #nn.Sigmoid()
            #nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Epoch:
    
    def __init__(self, autoencoder, device, dataloader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, loss_fn, optimizer,n=10):
        self.model = autoencoder
        self.device = device
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.avg_dataset = avg_dataset
        self.avg_dataset_test = avg_dataset_test
        self.avg = avg
        self.avg_test = avg_test
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n = n
        super().__init__()

    ### Training function
    def train_epoch(self,verbose=False):
        # Set train mode for both the encoder and the decoder
        train_loss = []

        for data in self.dataloader:
            inputs, label = data  # Assuming your dataloader provides (input, label) pairs
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
                avg_outputs_test = self.avg_dataset_test[:][0]
        
                avg_outputs_test = avg_outputs_test[label[:,1]-1-30,label[:,0]-1].to(self.device)

                outputs = self.model(inputs)

                outputs_device = outputs.to(self.device)

                loss = self.loss_fn(avg_outputs_test, outputs_device)  # Reconstruction loss
                test_loss.append(loss.item())

            return np.mean(test_loss)


    
    def train(self,num_epochs=2,verbose=False):
        self.diz_loss = {'train_loss':[],'val_loss':[]}
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(verbose=verbose)
            val_loss = self.test_epoch()
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
            self.diz_loss['train_loss'].append(train_loss)
            self.diz_loss['val_loss'].append(val_loss)
            #self.plot_ae_outputs()
        self.plot_losses()
        
    
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
        #test_transform = transforms.Compose([
        #transforms.ToTensor(),
        #])
        
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

def average(data,labels):
    avg = np.zeros((30,4,35,256))
    avg_labels = np.zeros((30,2))
    for i in range(1,5):
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

def average_2(data,labels):
    avg = np.zeros((10,4,35,256))
    avg_labels = np.zeros((10,2))
    for i in range(1,5):
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


def dataload():
    train_data = np.load("train_data_1_30.npy")
    train_labels = np.load("train_label_1_30.npy").astype(np.int32)
    test_data = np.load("test_data_31_40.npy")
    test_labels = np.load("test_label_31_40.npy").astype(np.int32)

    batch_size=32

    #train_data = np.transpose(train_data, (1,2,0))

    train_data = train_data*1e5
    test_data = test_data*1e5

    avg, avg_labels = average(train_data, train_labels)
    avg_test, avg_labels_test = average_2(test_data, test_labels)

    train_dataset = EEGDataset(train_data, train_labels)
    test_dataset = EEGDataset(test_data, test_labels)

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

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    m=len(train_dataset)

    #print(test_dataset[:,1])

    train_data, val_data = random_split(train_dataset, [math.floor(m*0.8), math.ceil(m*0.2)])
    #print(train_data[:5])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    return train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
autoencoder = EEGAutoencoder()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test = dataload()
dataloader = train_loader
num_epochs = 10

model = Epoch(autoencoder, device, train_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, criterion, optimizer, n=10)
    
#model.to(device)
model2 = model.train(num_epochs=num_epochs) #dataloader, loss_fn, optimizer,n=10))