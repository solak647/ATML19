import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torchvision import models
from torch.optim import lr_scheduler
class Model:

    def __init__(self, model = None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if model is None:
            model_ft = models.resnet18(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 10)
            model_ft = model_ft.to(self.device)
            self.model = model_ft
        else:
            self.model = model

    def train(self, train_loader, optimizer, loss_fn, print_every=100):
        '''
        Trains the model for one epoch
        '''
        self.model.train()
        losses = []
        n_correct = 0
        for iteration, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            output = self.model(images)
            optimizer.zero_grad()
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            n_correct += torch.sum(output.argmax(1) == labels).item()
        accuracy = 100.0 * n_correct / len(train_loader.dataset)
        return np.mean(np.array(losses)), accuracy
                
    def test(self, test_loader, loss_fn):
        '''
        Tests the model on data from test_loader
        '''
        self.model.eval()
        test_loss = 0
        n_correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.model(images)
                loss = loss_fn(output, labels)
                test_loss += loss.item()
                n_correct += torch.sum(output.argmax(1) == labels).item()

        average_loss = test_loss / len(test_loader)
        accuracy = 100.0 * n_correct / len(test_loader.dataset)
        return average_loss, accuracy


    def fit(self, train_dataloader, val_dataloader, optimizer, loss_fn, n_epochs=25, scheduler=None):
        if scheduler is None:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        best_val_loss = np.inf
        val_accuracy_best = 0
        best_model = None
        patience = 6 # if no improvement after 6 epochs, stop training
        counter = 0
        for epoch in range(n_epochs):
            train_loss, train_accuracy = self.train(train_dataloader, optimizer, loss_fn)
            val_loss, val_accuracy = self.test(val_dataloader, loss_fn)
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_accuracy_best = val_accuracy
                best_model = deepcopy(model)
                counter = 0
            else:
                counter += 1
            if counter == patience:
                print('No improvement for {} epochs; training stopped.'.format(patience))
                break
        
        return best_val_loss, val_accuracy_best, best_model


