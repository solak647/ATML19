import numpy as np
import torch

from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, Grayscale, ColorJitter
from torch.utils.data import DataLoader

import torch.nn as nn

from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 20
data_dir = 'data/'
root_dir = data_dir + 'train'

target_size = (100,100)
transforms = Compose([
                    Grayscale(num_output_channels=1),
                    ColorJitter(brightness=0.2,contrast=0.2),
                   # Resize(target_size), # Resizes image
                    ToTensor(),           # Converts to Tensor, scales to [0, 1] float (from [0, 255] int)
                    Normalize((0.5,), (0.5,)), # scales to [-1.0, 1.0]
                    ])

train_dataset = ImageFolder(root_dir, transform=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

# Same for validation dataset
val_root_dir = data_dir + 'val'
val_dataset = ImageFolder(val_root_dir, transform=transforms)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Same for test dataset
test_root_dir = data_dir + 'test'
test_dataset = ImageFolder(test_root_dir, transform=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# ADDING EARLY STOPPING
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(model, train_loader, optimizer, loss_fn, print_every=100):
    '''
    Trains the model for one epoch
    '''
    model.train()
    losses = []
    n_correct = 0
    for iteration, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
#         if iteration % print_every == 0:
#             print('Training iteration {}: loss {:.4f}'.format(iteration, loss.item()))
        losses.append(loss.item())
        n_correct += torch.sum(output.argmax(1) == labels).item()
    accuracy = 100.0 * n_correct / len(train_loader.dataset)
    return np.mean(np.array(losses)), accuracy
            
def test(model, test_loader, loss_fn):
    '''
    Tests the model on data from test_loader
    '''
    model.eval()
    test_loss = 0
    n_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            test_loss += loss.item()
            n_correct += torch.sum(output.argmax(1) == labels).item()

    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * n_correct / len(test_loader.dataset)
#     print('Test average loss: {:.4f}, accuracy: {:.3f}'.format(average_loss, accuracy))
    return average_loss, accuracy


def fit(train_dataloader, val_dataloader, model, optimizer, loss_fn, n_epochs, scheduler=None):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_loss = np.inf
    val_accuracy_best = 0
    best_model = None
    patience = 5 # if no improvement after 5 epochs, stop training
    counter = 0
    for epoch in range(n_epochs):
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_fn)
        val_loss, val_accuracy = test(model, val_dataloader, loss_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        if scheduler:
            scheduler.step() # argument only needed for ReduceLROnPlateau
        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                          train_losses[-1],
                                                                                                          train_accuracies[-1],
                                                                                                          val_losses[-1],
                                                                                                          val_accuracies[-1]))
        ### Early stopping code
        if val_accuracy > val_accuracy_best:
            best_val_loss = val_loss
            val_accuracy_best = val_accuracy
            best_model = deepcopy(model)
            counter = 0
        else:
            counter += 1
        if counter == patience and False:
            print('No improvement for {} epochs; training stopped.'.format(patience))
            break
    
    return best_val_loss, val_accuracy_best
    
class Conv1DNet2(nn.Module):
    
    def __init__(self):
        super(Conv1DNet2, self).__init__()
        self.conv = nn.Sequential(
          # input: 1x216x216
          nn.Conv2d(1, 128, (5,1)),
          # output: 128x212x216
          nn.BatchNorm2d(128,momentum=0.9),
          nn.LeakyReLU(0.2),
          nn.MaxPool2d(2),
          nn.Dropout(0.5),
          # output: 128x106x108
          nn.Conv2d(128, 64, (5,1)),
          # output: 64x102x108
          nn.BatchNorm2d(64,momentum=0.9),
          nn.LeakyReLU(0.2),
          nn.MaxPool2d(2),
          nn.Dropout(0.5),
          # output: 64x51x54
          nn.Conv2d(64, 64, (4,1)),
          # output: 64x48x54
          nn.BatchNorm2d(64,momentum=0.9),
          nn.LeakyReLU(0.2),
          nn.MaxPool2d(2),
          nn.Dropout(0.5),
          # output: 64x24x27
          nn.Conv2d(64, 64, (5,1)),
          # output: 64x20x27
          nn.BatchNorm2d(64,momentum=0.9),
          nn.LeakyReLU(0.2),
          nn.MaxPool2d((2,1), (2,1)),
          nn.Dropout(0.5)
          # output: 64x10x27
        )
        self.fc = nn.Sequential(
          nn.Linear(64*10*27,364),
          nn.LeakyReLU(0.2),
          nn.Dropout(0.5),
          nn.Linear(364,192),
          nn.LeakyReLU(0.2),
          nn.Dropout(0.5),
          nn.Linear(192,10)
        )
    
    def forward(self, input):
        output = self.conv(input)
        output = output.view(output.size(0), 64*10*27)
        output = self.fc(output)
        return output

model_conv = Conv1DNet2()
model_conv = model_conv.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(model_conv.parameters(), lr=learning_rate)
n_epochs = 50
loss_fn = nn.CrossEntropyLoss()

val_loss, val_accuracy = fit(train_dataloader, val_dataloader, model_conv, optimizer, loss_fn, n_epochs)