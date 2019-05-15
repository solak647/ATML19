import os
from os import listdir
from os.path import isfile, join
import random
from shutil import copyfile, rmtree

class DataFolder:
    
    def __init__(self, destination, remove_datafolder = False):
        
        self.directory = destination

        self.folders = ['train','val','test']
        self.styles = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        if remove_datafolder:
            if os.path.exists(self.directory):
                rmtree(self.directory)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        for folder in self.folders:
            if not os.path.exists(self.directory + '/' + folder):
                os.makedirs(self.directory + '/' + folder) 
            for style in self.styles:
                if not os.path.exists(self.directory + '/' + folder + '/' + style):
                    os.makedirs(self.directory + '/' + folder  + '/' + style) 
        pass
    
    def generate(self, spectrogram_directory, train_pourcent=0.6, val_pourcent=0.1):
        for style in self.styles:
            path = spectrogram_directory + '/' + style
            train_dest = self.directory + '/train/' + style
            val_dest = self.directory + '/val/' + style
            test_dest = self.directory + '/test/' + style
            files = [f for f in listdir(path) if isfile(join(path, f))]
            size = len(files)
            random.shuffle(files)
            train = files[: int(size * train_pourcent)]
            val = files[int(size * train_pourcent):int(size * (train_pourcent+val_pourcent))]
            test = files[int(size * (train_pourcent+val_pourcent)):]
            for file in train:
                copyfile(path + "/" + file, train_dest + "/" + file)
            for file in val:
                copyfile(path + "/" + file, val_dest + "/" + file)
            for file in test:
                copyfile(path + "/" + file, test_dest + "/" + file)