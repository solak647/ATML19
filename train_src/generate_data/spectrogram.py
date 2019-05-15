import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt
import urllib, base64
import pylab
import sys
import os
from PIL import Image
from io import BytesIO

class Spectrogram:
    
    def __init__(self, spectrogram_directory):
    
        # number of frequency bins (vertical resolution of the spectrogram)
        self.n_mels = 128
        
        self.n_fft=2048
        self.hop_length=512
        self.power=2.0
        
        # window size
        self.window_size = 100000
        self.window_step =  70000
        self.styles = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        self.directory = spectrogram_directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        for style in self.styles:
            if not os.path.exists(self.directory + '/' + style):
                os.makedirs(self.directory  + '/' + style)
        pass
    
    def create(self, wav_directory):
        minSamples = None
        for style in self.styles:
            for filename in os.listdir(wav_directory+"/"+style):
                if filename.endswith(".wav"):
                    sample_rate, samples = wavfile.read(wav_directory+'/'+style+'/'+filename)
                    if(minSamples == None or len(samples) < minSamples):
                        minSamples = len(samples)
        
        n_rolls = int((len(samples) - self.window_size) / self.window_step)
        print("The minimum number of samples is: ",minSamples)

        for style in self.styles:
            print("Working on "+style+" files")
            for filename in os.listdir(wav_directory+"/"+style):
                if filename.endswith(".wav"):
                    sample_rate, samples = wavfile.read(wav_directory+'/'+style+'/'+filename)
                    
                    # make the sample mono
                    samples = samples[:,0]

                    # normalize samples
                    samples = samples / np.max(np.abs(samples))

                    # split the sample    
                    splits = []
                    imgs = []
                    
                    for roll in range(n_rolls):
                        splits.append(samples[:self.window_size])
                        samples = np.roll(samples, -self.window_step)

                    for i, split in enumerate(splits):
                        S = librosa.feature.melspectrogram(y=split, sr=sample_rate,
                            n_mels=self.n_mels, n_fft=self.n_fft,
                            hop_length=self.hop_length, power=self.power)

                        pylab.figure(figsize=(3,3))
                        pylab.axis('off') 
                        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
                        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time', cmap="gray")
                        pylab.savefig(self.directory+'/'+style+'/'+filename+'-'+str(i)+'.png', bbox_inches=None, pad_inches=0)
                        pylab.clf()
                        pylab.close()
