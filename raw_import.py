import numpy as np
from mne.io import read_raw_eeglab, read_epochs_eeglab
import matplotlib.pyplot as plt
import os
import pandas as pd

#PATH = os.path.dirname(os.path.abspath(__file__)) # save this script in same directory as EEG folder

#E = 33 # 0-32 electrodes
#N = 699392 # number of data points
#DT = 0.0009765625 # time-period between each data point

subject = 1 # 1-40

class EEG():
    
    def __init__(self):
        self.subject_n = 40
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


    def get_epoch_data(self,subject=1, folder="N170 Raw Data and Scripts Only", file="_N170.set"):
        filename = self.PATH + '/' + folder + '/' + str(subject) + '/' + str(subject)+file

        epoch_data = read_epochs_eeglab(filename)

        for i, item in enumerate(epoch_data):
            print(item[0,1000])

        #print(epoch_data)

    def preprocess(self):
        return 0


    def std_normalize(self):
        return 0


    def mm_normalize(self,raw_matrix): ##Consider how to normalize data (e.g. over each electrode, over each patient etc.)

        
        a_min = raw_matrix.min(axis=(1), keepdims=True)
        a_max = raw_matrix.max(axis=(1), keepdims=True)
        a_norm = (raw_matrix - a_min)/(a_max - a_min)

        return a_norm
    


    def plotting(self):
        return 0

    def data_split(self, data):
        x, y = np.rollaxis(data, axis=2)
        return x,y
    
    def data_to_csv(self, data, filename="raw_data"):
        full_filename_1 = self.PATH + "/" + filename + "_x.csv"
        full_filename_2 = self.PATH + "/" + filename + "_y.csv"

        x,y = self.data_split(data)

        pd.DataFrame(x).to_csv(full_filename_1)
        pd.DataFrame(y).to_csv(full_filename_2)
        print("data saved as csv")

    def electrode_avg(self, data):
        x,y = self.data_split(data)

        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y, axis=0)

        return x_mean, y_mean
    

eeg = EEG()
raw_data = eeg.get_raw_data(subject=subject, folder="N170 Raw Data and Scripts Only", file="_N170.set")
raw_x_mean, raw_y_mean = eeg.electrode_avg(raw_data)

shifted_data = eeg.get_raw_data(subject=subject, folder="N170 Raw Data and Scripts Only", file="_N170_shifted.set")
shifted_x_mean, shifted_y_mean = eeg.electrode_avg(shifted_data)

shifted_ds_data = eeg.get_raw_data(subject=subject, folder="N170 Raw Data and Scripts Only", file="_N170_shifted_ds.set")
shifted_ds_x_mean, shifted_ds_y_mean = eeg.electrode_avg(shifted_ds_data)

shifted_ds_reref_ucbip_data = eeg.get_raw_data(subject=subject, folder="N170 Raw Data and Scripts Only", file="_N170_shifted_ds_reref_ucbip.set")
shifted_ds_reref_ucbip_x_mean, shifted_ds_reref_ucbip_y_mean = eeg.electrode_avg(shifted_ds_reref_ucbip_data)

shifted_ds_reref_ucbip_hpfilt_data = eeg.get_raw_data(subject=subject, folder="N170 Raw Data and Scripts Only", file="_N170_shifted_ds_reref_ucbip_hpfilt.set")
shifted_ds_reref_ucbip_hpfilt_x_mean, shifted_ds_reref_ucbip_hpfilt_y_mean = eeg.electrode_avg(shifted_ds_reref_ucbip_hpfilt_data)




#eeg.get_epoch_data(subject=subject, folder="N170 Raw Data and Scripts Only", file="_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set")

#print(final_data.shape[:])
#norm = eeg.mm_normalize(raw_matrix)



#fig, axs = plt.subplots(3,1)
    #print(i)


axs[0].plot(raw_data[0,:,0], raw_data[0,:,1], color="b")
#axs[0].plot(raw_x_mean, raw_y_mean, color="r")

axs[1].plot(shifted_ds_reref_ucbip_data[0,:,0], shifted_ds_reref_ucbip_data[0,:,1], color="b")
#axs[1].plot(shifted_ds_reref_ucbip_x_mean, shifted_ds_reref_ucbip_y_mean, color="r")

axs[2].plot(shifted_ds_reref_ucbip_hpfilt_data[0,:,0], shifted_ds_reref_ucbip_hpfilt_data[0,:,1], color="b")
plt.show()

