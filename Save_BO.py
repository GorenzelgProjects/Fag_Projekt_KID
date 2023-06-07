
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

class SaveData:
    def __init__(self, architecture=[],hyperparameter=[],encoder=None,decoder=None,plots=None, folder_name=None, relative_path="./data"):
        self.architecture = architecture
        self.hyperparameter = hyperparameter
        self.encoder = encoder
        self.decoder = decoder
        self.plots = plots
        self.folder_name = folder_name
        self.relative_path = relative_path
    
    def save_data_to_txt(self,save_architecture=False):
        # If dict doesn't exist, create new folder
        if self.folder_name:
            directory = os.path.join(self.relative_path, self.folder_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
        else:
            directory = self.relative_path

        # Create a new .txt at the directory
        file_name = "data.txt"
        file_path = os.path.join(directory, file_name)

        with open(file_path, "w") as f:
            # Write each item in the list to a new line in the file
            f.write(str("Architecture:")+ "\n" + "\n")
            for item in self.architecture:
                f.write(str(item) + "\n")
            f.write(str("Hyperparameters:")+ "\n" + "\n")
            for item in self.hyperparameter:
                f.write(str(item) + "\n")

        if save_architecture==True:
            file_name = ["decoder.pt","encoder.pt"]
            file_path = [os.path.join(directory, file_name[0]),os.path.join(directory, file_name[1])]
            torch.save(self.encoder.state_dict(), file_path[0])
            torch.save(self.decoder.state_dict(), file_path[1])

        
        print(f"Architecture and hyperparameters saved to {file_path}.txt")

    def save_plots(self):
        if self.folder_name:
            directory = os.path.join(self.relative_path, self.folder_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
        else:
            directory = self.relative_path

        # Create a new .txt at the directory
        file_name = "plot.png"
        file_path = os.path.join(directory, file_name)

        plt.savefig(file_path)
        plt.close()

        print(f"plot saved to {file_path}.png")
        return ##This is under revision as of what is smartest
        
if __name__ == "__main__":
# Example usage:
    save = SaveData()
    my_data = ["apple", "banana", "cherry", "date"]
    save.save_data_to_txt(my_data, folder_name="fruits", relative_path="./data_files")