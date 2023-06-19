import numpy as np
from mne.io import read_raw_eeglab, read_epochs_eeglab
import os
import pandas as pd
import re


# The class EEG is made to import data from ERP-CORE and sort in a more usuable way.
class EEG():
    
    def __init__(self):
        self.subject_n = 40
        self.E = 33 # 0-32 electrodes
        self.N = 699392 # number of data points
        self.DT = 0.0009765625 # time-period between each data point
        #self.PATH = os.path.dirname(os.path.abspath(__file__)) # save this script in same directory as EEG folder
        self.PATH = "H:/EEG"

    # Function to import raw EEG imports and return as matrix
    def get_raw_data(self,subject=1, folder="N170 Raw Data and Scripts Only", file="_N170.set"):
        filename = self.PATH + '/' + folder + '/' + str(subject) + '/' + str(subject)+file
        
        raws = read_raw_eeglab(filename, preload=True)

        x, y = [], []

        for i in range(self.E):
            x.append(raws[i][1])
            y.append(raws[i][0][0])

        raw_matrix = np.dstack((x,y))

        return raw_matrix #returns 3d array (E,N,(x,y))

    # Function to import event related epoch trials, sort them in correctly in bins and subject with represented labels, and return a matrix with data and a matrix with labels.
    def get_epoch_data(self,subject=1, folder="N170 Raw Data and Scripts Only", file="_N170.set", total_bins = 0):
        filename = self.PATH + '/' + folder + '/' + str(subject) + '/' + str(subject)+file
        bin_sub = []
        epoch_data = read_epochs_eeglab(filename, verbose=False)
        #print(len(epoch_data.events))
        #print(epoch_data.event_id)
        bins = []
        events = epoch_data.events
        list_bin = list(epoch_data.event_id.keys())
        #print(list_bin)
        for i in events[:,-1]:
            #print(i)
            idx = list_bin[i-1].index("B")
            bins.append(int(list_bin[i-1][idx+1]))
        #bins = bins[:,-1]
        dif = int(min(bins) - 1)
        #dif = 0
        if 0 not in bins:
            unique_bins = len(set(bins))
        else:
            unique_bins = len(set(bins)) - 1

        for b in bins:
            if b != 0:
                bin_sub.append([b-dif+total_bins,subject])
        n_bins = len(epoch_data)
        #print(n_bins)
        data = np.zeros([n_bins,35,256])
        for i, item in enumerate(epoch_data):
            #print(epoch_data[i].event_id)
            data[i,:,:] = item

        return data, bin_sub, unique_bins

    # Function to get bins given an event related code index.
    def get_bins(self, subject=1, folder="N170 Raw Data and Scripts Only", file="_N170_Eventlist_Bins.txt", total_bins = 0):
        filename = self.PATH + '/' + folder + '/' + str(subject) + '/' + str(subject) + file
        f = open(filename, "r")
        text = f.readlines()[20:]
        bin_list = []
        unique_bins = []
        for line in text:
            start_index = str(line).find("[")+1
            end_index = str(line).find("]")
            if end_index == -1:
                end_index = 0
            temp_bin = line[start_index:end_index].strip()
            if len(temp_bin) > 0:
                if len(temp_bin) > 1:
                    temp_bin = int(temp_bin[-1])-2
                temp_bin = int(temp_bin) + total_bins
                bin_list.append([temp_bin, subject])
                if temp_bin not in unique_bins:
                    unique_bins.append(temp_bin)
        return bin_list, len(unique_bins)  

eeg = EEG()
subject = 1 # 1-40


# list with paradigm folders used
folder_list = ["N170 Raw Data and Scripts Only",
               "MMN All Data and Scripts",
               "N2pc All Data and Scripts",
               "P3 All Data and Scripts",
               "N400 All Data and Scripts"]

# list with paradigm datapreprocess files used
file_list = ["_N170_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set",
             "_MMN_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set",
             "_N2pc_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set",
             "_P3_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set",
             "_N400_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp.set"]

# list with paradigms event list used
bin_list = ["_N170_Eventlist_Bins.txt",
            "_MMN_Eventlist_Bins.txt",
            "_N2pc_Eventlist_Bins.txt",
            "_P3_Eventlist_Bins.txt",
            "_N400_Eventlist_Bins.txt"]


# boolean variables to indicate if first paradigm call
first_train = True
first_test = True

total_bins = 0
shape_list = []

# looping over paradigms folder,data-files, and eventlist-files
for folder, file, bin in zip(folder_list, file_list, bin_list):
    print(folder)
    unique_bin = 0

    #looping over each subject for train data
    for sub in range(1,31):
        #Extracting event related epochs for a given subject
        data, bin_list, unique_bins = eeg.get_epoch_data(subject=sub, folder=folder, file=file, total_bins = total_bins)

        if first_train:
            all_data = np.copy(data)
            all_labels = np.copy(bin_list)
            first_train = False
        else:
            all_data = np.vstack((all_data, data))
            all_labels = np.vstack((all_labels, bin_list))

        if unique_bins > unique_bin:
            unique_bin = unique_bins

    #looping over each subject for test data
    for sub in range(31,41):

        #Extracting event related epochs for a given subject
        data, bin_list, unique_bins = eeg.get_epoch_data(subject=sub, folder=folder, file=file, total_bins = total_bins)


        if first_test:
            all_data_test = np.copy(data)
            all_labels_test = np.copy(bin_list)
            first_test = False
        else:
            all_data_test = np.vstack((all_data_test, data))
            all_labels_test = np.vstack((all_labels_test, bin_list))

        if unique_bins > unique_bin:
            unique_bin = unique_bins

    total_bins += unique_bin

# Saving train data and label, test data and labels as numpy files.
np.save("train_data_1_30", all_data)
np.save("train_label_1_30", all_labels)
np.save("test_data_31_40", all_data_test)
np.save("test_label_31_40", all_labels_test)


# Printing shapes to evaluate sizes.
print(all_data.shape[:])
print(all_labels.shape[:])
print(all_data_test.shape[:])
print(all_labels_test.shape[:])
print(total_bins

