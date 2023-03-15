import numpy as np
from mne.io import read_raw_eeglab
import matplotlib.pyplot as plt
import os


#PATH = os.path.dirname(os.path.abspath(__file__)) # save this script in same directory as EEG folder

#E = 33 # 0-32 electrodes
#N = 699392 # number of data points
#DT = 0.0009765625 # time-period between each data point

subject = 1 # 1-40

class EEG():
    
    def __init__(self):
        self.E = 33 # 0-32 electrodes
        self.N = 699392 # number of data points
        self.DT = 0.0009765625 # time-period between each data point
        self.PATH = os.path.dirname(os.path.abspath(__file__)) # save this script in same directory as EEG folder

    def get_raw_data(self,subject=1, folder="N170 Raw Data and Scripts Only", file="_N170.set"):
        filename = self.PATH + '/' + folder + '/' + str(subject) + '/' + str(subject)+file
        
        raws = read_raw_eeglab(filename, preload=True)

        x, y = [], []

        for i in range(self.E):
            x.append(raws[i][1])
            y.append(raws[i][0][0])

        raw_matrix = np.dstack((x,y)) 

        return raw_matrix #returns 3d array (E,N,(x,y))


    def preprocess(self):
        return 0


    def std_normalize(self):
        return 0


    def mm_normalize(self,raw_matrix):

        
        a_min = raw_matrix.min(axis=(1), keepdims=True)
        a_max = raw_matrix.max(axis=(1), keepdims=True)
        a_norm = (raw_matrix - a_min)/(a_max - a_min)

        return a_norm
    


    def plotting(self):
        return 0
    
    

eeg = EEG()
raw_matrix = eeg.get_raw_data(subject=subject, folder="N170 Raw Data and Scripts Only", file="_N170.set")
print(raw_matrix.shape[:])

norm = eeg.mm_normalize(raw_matrix)

print(norm.shape)


#print(raw_matrix[0,:100,0])


#print(raw_matrix[0][1000][0])
#print(raw_matrix[0][1000][1])
fig, axs = plt.subplots(3,1)
    #print(i)

#for i in range(100):
    #print(raw_matrix[0][i][0])
    #print(raw_matrix[0][i][1])

axs[0].plot(raw_matrix[0,:,0], raw_matrix[0,:,1], color="b")
plt.show()
