import torch.nn as nn
import torch
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, Grayscale, ColorJitter
from torch.autograd import Variable


class Conv1DNet2(nn.Module):
    
    def __init__(self):
        super(Conv1DNet2, self).__init__()
        
        self.transforms = Compose([
                        Grayscale(num_output_channels=1),
                        ToTensor(),           # Converts to Tensor, scales to [0, 1] float (from [0, 255] int)
                        Normalize((0.5,), (0.5,)), # scales to [-1.0, 1.0]
                      ])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
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
        
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def predict_image(self, image):
        image.save('out.png')
        image_tensor = self.transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = nn.Softmax()(self(input))
        index = output.data.cpu().numpy()
        return index
            
