import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import copy
from torch.utils.data import Dataset, DataLoader
import math

class NoisyDataSetModel(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target
        self.n_samples = self.input.shape[0]

    def __getitem__(self, index):
        return self.input[index], self.target[index]

    def __len__(self):
        return self.n_samples


class ResUNet_n2n(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ResUNet_n2n, self).__init__()

        #implementing the residual blocks without concatenation (this is in the forward pass)
        self.rb1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1), #from in_channels to 48 out channels, with 3x3 kernel size
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, stride=1, padding=1),  #going from 48 to 48 channels/features
            nn.ReLU(),
            nn.MaxPool2d(2) #kernel size 2x2
        )

        #four equal blocks: block 2 to 5
        self.rb2_5 = nn.Sequential(
            nn.Conv2d(48,48, 3, stride=1, padding=1), #48 to 48 features, 3x3 kernel
            nn.ReLU(),
            nn.MaxPool2d(2) #max pooling by 2x2 kernel
        )

        self.rb6 = nn.Sequential(
            nn.Conv2d(48,48,3, stride=1, padding=1), #48 to 48 featrues, 3x3 kernel
            nn.ReLU(),
            nn.ConvTranspose2d(48,48,kernel_size=3, stride=2, padding=1, output_padding=1) #this is equal to bilinear upscaling with factor 2 and then applying nn.Conv2d(48,48,3,padding=1)
            #see lecture 7.1 "transposed convolutions", slide 46 --> because of stride=2.
            #Important: number of channels remain the same - this will be changed by adding the skip connections.
        )

        #doubled the feature channel with the upsampling
        self.rb7 = nn.Sequential(
            nn.Conv2d(96,96,3, stride=1, padding=1), #96 to 96 featrues, 3x3 kernel
            nn.ReLU(),
            nn.Conv2d(96,96,3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96,96,kernel_size=3, stride=2, padding=1, output_padding=1) #upsampling by 2x2
            #Important: number of channels remain the same - this will be changed by adding the skip connections.
        )

        #three times the same block 8 - 10
        self.rb8_10 = nn.Sequential(
            nn.Conv2d(144,96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(96,96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96,96,kernel_size=3, stride=2, padding=1, output_padding=1) #upsampling by 2x2
            #Important: number of channels remain the same - this will be changed by adding the skip connections.
        )

        self.rb11 = nn.Sequential(
            nn.Conv2d(in_channels + 96, 64, kernel_size=3, stride=1, padding=1), #reducing channels, kernel 3x3
            nn.ReLU(),
            nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1 ), #reducing channels, kernel 3x3
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU() #Maas et al, 2013, alpha = 0.1. stated in paper (see at the top)
        )

    #computing the pooling outputs
    def forward(self, x):
        # -------- encoder ----------
        p1 = self.rb1(x)
        p2 = self.rb2_5(p1)
        p3 = self.rb2_5(p2)
        p4 = self.rb2_5(p3)
        p5 = self.rb2_5(p4)

        # -------- decoder ----------
        y = self.rb6(p5) #result is upsampled
        cat1 = torch.cat((y, p4), dim=1) #https://pytorch.org/docs/stable/generated/torch.cat.html --> number of samples n stays the same
        y = self.rb7(cat1) #96 channels input to rb7, result will be upsampled
        cat2 = torch.cat((y, p3), dim=1)
        y = self.rb8_10(cat2) #144 channels input to rb8_10, result will be upsampled
        cat3 = torch.cat((y, p2), dim=1)
        y = self.rb8_10(cat3) #144 channels input to rb8_10, result will be upsampled
        cat4 = torch.cat((y, p1), dim=1)
        y = self.rb8_10(cat4) #144 channels input to rb8_10, result will be upsampled
        cat5 = torch.cat((y, x), dim=1) #concatenate with input
        return self.rb11(cat5) #leaky relu with alpha 0.1 is applied in rb11



class Model():

    def __init__(self):
        self.model = ResUNet_n2n()
        self.learning_rate = 0.001
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate) #betas=beta
        self.criterion = nn.MSELoss()
        self.mini_batch_size = 8

    def load_pretrained_model(self):
        self.model.load_state_dict(torch.load('./Miniproject_1/bestmodel.pth'))

    def train(self, train_input, train_target, num_epochs):

        if torch.cuda.is_available()==True:
            device="cuda:0"
        else:
            device ="cpu"

        train_input = (train_input).float()/255.0
        train_target = (train_target).float()/255.0

        tr_dataset = NoisyDataSetModel(train_input, train_target)
        dataloader = DataLoader(tr_dataset,  self.mini_batch_size, shuffle=True, num_workers=0)
        dataset_size = len(tr_dataset)

        model = self.model.to(device)
        optimizer = self.optimizer
        criterion = self.criterion

        for epoch in range(num_epochs):

            model.train()

            running_loss = 0.0

            for i, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                  outputs = model(inputs)
                  loss = criterion(outputs, targets)
                  loss.backward()
                  optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss


        self.model.load_state_dict(model.state_dict())


    def predict(self, input):
        if torch.cuda.is_available()==True:
            device="cuda:0"
        else:
            device ="cpu"

        input = input.float()/input.max()
        input = input.to(device)
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
          return self.model(input)*255.0
