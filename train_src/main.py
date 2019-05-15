from generate_data.spectrogram import Spectrogram
from generate_data.data_folder import DataFolder
from model.dataloader import DataLoader
from model.model import Model
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
import sys
import os

def main():

    # Do you want to generate the data folder from wav file? If yes, you need to have wav files.
    generate_data = False
    wav_directory = '../genres_wav'
    spectrogram_directory = '../genres_spectrograms'
    data_directory = 'data'
    if generate_data:
        
        # Create spectrogram from wav files
        spectrogram = Spectrogram(spectrogram_directory)
        spectrogram.create(wav_directory)

        # Create data folder with train, test and val directory
        data_folder = DataFolder(data_directory, False)
        data_folder.generate(spectrogram_directory, train_pourcent=0.6, val_pourcent=0.1)

    if not os.path.exists(data_directory):
        print('no data directory')
        sys.exit(0)
    print("Creating dataloader")
    dataloaders = DataLoader(data_directory, batch_size = 100)
    train_dataloader, val_dataloader, test_dataloader = dataloaders.get_dataloader()
    print("Creating the model")
    model = Model()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
    n_epochs = 50
    loss_fn = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
    print("Start learning")
    val_loss, val_accuracy, best_model = model.fit(train_dataloader, val_dataloader, optimizer, loss_fn, n_epochs, exp_lr_scheduler)
    print("End of learning")
    best = Model(best_model)
    loss_cnn, accuracy_cnn = best.test(test_dataloader,loss_fn)
    print('Test loss: {:.4f}, test accuracy: {:.4f}'.format(loss_cnn,accuracy_cnn))
    torch.save(best_model.state_dict(), 'models/best_model')
if __name__ == '__main__':
	main()
