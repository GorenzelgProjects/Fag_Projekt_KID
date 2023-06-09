import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split, Dataset
import numpy as np
from torchvision import transforms
import math
import matplotlib.pyplot as plt

class AET(nn.Module):
    def __init__(self,encoder_list=0,decoder_list=0, transformer_in=0, in_channels=[35,64], out_channels=[64,128], nhead=8, layers=3, kernel = 3, kernel_p = 2, stride = 1, stride_p = 2,padding = 1, padding_p = 0, pooling = True):
        super(AET, self).__init__()

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

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=transformer_in, nhead=nhead, dim_feedforward=out_channels[1])
        
        self.transformer = nn.Sequential(
            nn.TransformerEncoder(self.transformer_encoder, num_layers=3)
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
        transformed = self.transformer(encoded)
        decoded = self.decoder(transformed)
        return decoded

class Epoch:
    
    def __init__(self, autoencoder, device, dataloader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, loss_fn, optimizer,test_dataset, test_labels,n=10):
    #def __init__(self, autoencoder, device, dataloader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, loss_fn, optimizer,n=10):
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
        self.test_dataset = test_dataset
        self.test_labels = test_labels
        self.n = n
        super().__init__()

    ### Training function
    def train_epoch(self,verbose=False):
        # Set train mode for both the encoder and the decoder
        train_loss = []

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
        label_name = []

        targets = self.test_labels[:,0]
        t_idx = {1:np.where(targets==1)[0]}
        val = t_idx[1][0]
        img = self.test_dataset[val][0].unsqueeze(0)

        avg_outputs_test = self.avg_dataset_test[:][0]
        avg_outputs_norm = avg_outputs_test[:,0]
        

        for i in range(len(avg_outputs_norm[0])):
            avg_outputs_norm[0,i] = avg_outputs_norm[0,i] / np.linalg.norm(avg_outputs_norm[0,i],ord=1)

        avg_outputs_test = avg_outputs_norm[0]
        print(avg_outputs_test)
        with torch.no_grad():
            rec_img  = self.model(img.to(self.device))
        count =0
        rec = rec_img.cpu().squeeze().numpy()
        #print(rec.size())
        for i in range(len(rec)):
            rec[i] = rec[i] / np.linalg.norm(rec[i],ord=1)
            count += 1
        print(count)
        print(rec)


        x = np.arange(0,256)
        for i in range(35):
            displacement = i*0.05

            plt.plot(x,avg_outputs_test[i,:]+displacement, color='black')
            plt.plot(x,rec[i,:]+displacement, color='tomato', alpha=0.7)
            label_pos.append(avg_outputs_test[i,:].mean()+displacement)
            temp_name = "EEG " + str(i+1)
            label_name.append(temp_name)

        plt.yticks(label_pos, label_name)
        plt.xlim([0, 256])
        #plt.yticks(y, labels, rotation='vertical')
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

    return train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, test_dataset, test_labels
    #return train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test

if __name__ == "__main__":
    in_channels = [35,64]
    out_channels= [64,128]

    layers=2
    kernel = 3
    kernel_p = 2
    stride = 1
    stride_p = 2
    padding = 1
    padding_p = 0
    pooling = True
    nhead = 8
    num_epochs = 3
    
    encoder_list, decoder_list, transformer_in = calc_convolution(layers=layers, kernel = kernel, kernel_p = kernel_p, stride = stride, stride_p = stride_p, padding = padding, padding_p = padding_p, pooling = pooling) #Stride can't be change do to BO
    print(encoder_list, decoder_list)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    autoencoder = AET(encoder_list=encoder_list,decoder_list=decoder_list, transformer_in=transformer_in, in_channels=in_channels, out_channels=out_channels, nhead=nhead, layers=layers, kernel = kernel, kernel_p = kernel_p, stride = stride, stride_p = stride_p,padding = padding, padding_p = padding_p, pooling = pooling)
    autoencoder.to(device)
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    #criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.01, momentum=0.9)

    train_loader, valid_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, test_dataset, test_labels = dataload()
    dataloader = train_loader
    

    model = Epoch(autoencoder, device, train_loader, test_loader, avg_dataset, avg_dataset_test, avg, avg_test, criterion, optimizer, test_dataset, test_labels, n=5)   
    #model.to(device)
    model2 = model.train(num_epochs=num_epochs) #dataloader, loss_fn, optimizer,n=10))