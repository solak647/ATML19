import tkinter as tk
from tkinter import filedialog
from tkinter import *
from model import Model
from spectrogram import Spectrogram

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        
        self.spectrogram = Spectrogram()
        self.model = Model()
        
        self.musicFilePath = StringVar()
        self.defaultDir = "/home/peclatj/data/Document/Université/Master MCS/Advanced Topics in Machine Learning/ATML19/genres_wav/classical"


        self.create_widgets()        


    def create_widgets(self):
        self.chooseFileBtn = tk.Button(self)
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

        self.quit = tk.Button(self, text="Quit", fg="red", command=self.master.destroy)
        self.quit.pack(side="bottom")

    def chooseFile(self):
        filetypes = (("wave files","*.wav"),("all files","*.*"))
        path = filedialog.askopenfilename(initialdir = self.defaultDir, title="Select a music", filetypes=filetypes)
        self.musicFilePath.set(path)

    def classify(self):
        imgs = self.spectrogram.sample(self.musicFilePath.get())
        
        self.model.load("best_model_resnet")
        
        for img in imgs:
            print(self.model.predict_image(img))



root = tk.Tk()
app = Application(master=root)
app.mainloop()
