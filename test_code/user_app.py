import tkinter as tk
from tkinter import filedialog
from tkinter import *
from models.model import Model
import numpy as np
import torch.nn as nn
from generate_data.spectrogram import Spectrogram

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        
        self.spectrogram = Spectrogram()
        self.model = Model()
        
        self.numberGenre = 10
        self.musicFilePath = StringVar()
        self.bluesResult = StringVar()
        self.classicalResult = StringVar()
        self.countryResult = StringVar()
        self.discoResult = StringVar()
        self.hiphopResult = StringVar()
        self.jazzResult = StringVar()
        self.metalResult = StringVar()
        self.popResult = StringVar()
        self.reggaeResult = StringVar()
        self.rockResult = StringVar()
        
        self.defaultDir = "/home/peclatj/data/Document/Université/Master MCS/Advanced Topics in Machine Learning/ATML19/genres_wav/classical"

        self.create_widgets()


    def create_widgets(self):
        self.chooseFileBtn = tk.Button(self, width=200)
        self.chooseFileBtn["text"] = "ChooseFile"
        self.chooseFileBtn["command"] = self.chooseFile
        self.chooseFileBtn.pack(side="top")

        self.filePathPan = PanedWindow(self, orient=HORIZONTAL, width=1000)
        self.filePathPan.pack(side="top")
        
        self.filePath1Lbl = tk.Label(self.filePathPan, text="File path:")
        self.filePathPan.add(self.filePath1Lbl)
        
        self.filePath2Lbl = tk.Label(self.filePathPan, textvariable=self.musicFilePath)
        self.filePathPan.add(self.filePath2Lbl)
        
        self.classifyBtn = tk.Button(self)
        self.classifyBtn["text"] = "Classify"
        self.classifyBtn["command"] = self.classify
        self.classifyBtn.pack(side="top")
        
        # results
        self.resultsPan = PanedWindow(self, orient=VERTICAL, height=600)
        
        # upper results
        self.subResultsPan1 = PanedWindow(self.resultsPan, orient=HORIZONTAL, width=500)
        self.subResultsPan1.add(self.resultsPan)
        
        self.bluesPan = PanedWindow(self.subResultsPan1, orient=HORIZONTAL, width=100)
        self.blues1Lbl = tk.Label(self.bluesPan, text="Blues:")
        self.bluesPan.add(self.blues1Lbl)
        self.blues2Lbl = tk.Label(self.bluesPan, textvariable=self.bluesResult)
        self.bluesPan.add(self.blues2Lbl)
        self.subResultsPan1.add(self.bluesPan)
        
        self.classicalPan = PanedWindow(self.subResultsPan1, orient=HORIZONTAL, width=100)
        self.classical1Lbl = tk.Label(self.classicalPan, text="Classical:")
        self.classicalPan.add(self.classical1Lbl)
        self.classical2Lbl = tk.Label(self.classicalPan, textvariable=self.classicalResult)
        self.classicalPan.add(self.classical2Lbl)
        self.subResultsPan1.add(self.classicalPan)
        
        self.countryPan = PanedWindow(self.subResultsPan1, orient=HORIZONTAL, width=100)
        self.country1Lbl = tk.Label(self.countryPan, text="Country:")
        self.countryPan.add(self.country1Lbl)
        self.country2Lbl = tk.Label(self.countryPan, textvariable=self.countryResult)
        self.countryPan.add(self.country2Lbl)
        self.subResultsPan1.add(self.countryPan)
        
        self.discoPan = PanedWindow(self.subResultsPan1, orient=HORIZONTAL, width=100)
        self.disco1Lbl = tk.Label(self.discoPan, text="Disco:")
        self.discoPan.add(self.disco1Lbl)
        self.disco2Lbl = tk.Label(self.discoPan, textvariable=self.discoResult)
        self.discoPan.add(self.disco2Lbl)
        self.subResultsPan1.add(self.discoPan)
        
        self.hiphopPan = PanedWindow(self.subResultsPan1, orient=HORIZONTAL, width=100)
        self.hiphop1Lbl = tk.Label(self.hiphopPan, text="Hip-hop:")
        self.hiphopPan.add(self.hiphop1Lbl)
        self.hiphop2Lbl = tk.Label(self.hiphopPan, textvariable=self.hiphopResult)
        self.hiphopPan.add(self.hiphop2Lbl)
        self.subResultsPan1.add(self.hiphopPan)
        
        self.resultsPan.add(self.subResultsPan1)
        
        # lower results
        self.subResultsPan2 = PanedWindow(self.resultsPan, orient=HORIZONTAL, width=500)
        self.subResultsPan2.add(self.resultsPan)
        
        self.jazzPan = PanedWindow(self.subResultsPan2, orient=HORIZONTAL, width=100)
        self.jazz1Lbl = tk.Label(self.jazzPan, text="Jazz:")
        self.jazzPan.add(self.jazz1Lbl)
        self.jazz2Lbl = tk.Label(self.jazzPan, textvariable=self.jazzResult)
        self.jazzPan.add(self.jazz2Lbl)
        self.subResultsPan2.add(self.jazzPan)
        
        self.metalPan = PanedWindow(self.subResultsPan2, orient=HORIZONTAL, width=100)
        self.metal1Lbl = tk.Label(self.metalPan, text="Metal:")
        self.metalPan.add(self.metal1Lbl)
        self.metal2Lbl = tk.Label(self.metalPan, textvariable=self.metalResult)
        self.metalPan.add(self.metal2Lbl)
        self.subResultsPan2.add(self.metalPan)
        
        self.popPan = PanedWindow(self.subResultsPan2, orient=HORIZONTAL, width=100)
        self.pop1Lbl = tk.Label(self.popPan, text="Pop:")
        self.popPan.add(self.pop1Lbl)
        self.pop2Lbl = tk.Label(self.popPan, textvariable=self.popResult)
        self.popPan.add(self.pop2Lbl)
        self.subResultsPan2.add(self.popPan)
        
        self.reggaePan = PanedWindow(self.subResultsPan2, orient=HORIZONTAL, width=100)
        self.reggae1Lbl = tk.Label(self.reggaePan, text="Reggae:")
        self.reggaePan.add(self.reggae1Lbl)
        self.reggae2Lbl = tk.Label(self.reggaePan, textvariable=self.reggaeResult)
        self.reggaePan.add(self.reggae2Lbl)
        self.subResultsPan2.add(self.reggaePan)
        
        self.rockPan = PanedWindow(self.subResultsPan2, orient=HORIZONTAL, width=100)
        self.rock1Lbl = tk.Label(self.rockPan, text="Rock:")
        self.rockPan.add(self.rock1Lbl)
        self.rock2Lbl = tk.Label(self.rockPan, textvariable=self.rockResult)
        self.rockPan.add(self.rock2Lbl)
        self.subResultsPan2.add(self.rockPan)
        
        
        self.resultsPan.add(self.subResultsPan2)
        
        self.resultsPan.pack(side="bottom")
        
        self.quit = tk.Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")

    def chooseFile(self):
        filetypes = (("wave files","*.wav"),("all files","*.*"))
        path = filedialog.askopenfilename(initialdir = self.defaultDir, title="Select a music", filetypes=filetypes)
        self.musicFilePath.set(path)

    def classify(self):
        results_sum = np.array([0.0] * self.numberGenre)
        
        #self.model.load("best_model")
        imgs = self.spectrogram.sample(self.musicFilePath.get())
        self.model.load("models/best_model_resnet")

        for img in imgs:
            results_sum += self.model.predict_image(img)

        x = results_sum / len(imgs)
        y = x / sum(x) * 100.0
        
        self.bluesResult.set(str(int(y[0])) + "%")
        self.classicalResult.set(str(int(y[1])) + "%")
        self.countryResult.set(str(int(y[2])) + "%")
        self.discoResult.set(str(int(y[3])) + "%")
        self.hiphopResult.set(str(int(y[4])) + "%")
        self.jazzResult.set(str(int(y[5])) + "%")
        self.metalResult.set(str(int(y[6])) + "%")
        self.popResult.set(str(int(y[7])) + "%")
        self.reggaeResult.set(str(int(y[8])) + "%")
        self.rockResult.set(str(int(y[9])) + "%")
        

root = tk.Tk()
app = Application(master=root)
app.mainloop()
