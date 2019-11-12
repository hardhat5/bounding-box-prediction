# Importing relevant libraries
import os
import torch
import torch.nn as nn
import pandas as pd
import skimage
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
import csv
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.0001
batch_size = 4

# Dataset loader
class ProductsDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.products_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.products_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.products_frame.iloc[idx, 0])
        image = io.imread(img_name)
        image = skimage.transform.resize(image, (224,224))
        x1 = self.products_frame.iloc[idx, 1]
        x2 = self.products_frame.iloc[idx, 2]
        y1 = self.products_frame.iloc[idx, 3]
        y2 = self.products_frame.iloc[idx, 4]

        sample = {"image": image, 'coordinates': np.array([x1,x2,y1,y2])}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Transforming dataset to dictionary
class ToTensor(object):
    def __call__(self, sample):
        image, coordinates = sample['image'], sample['coordinates']
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image)
        coordinates = torch.from_numpy(coordinates)
        return {'image': image.float(),
               'coordinates': coordinates.float()}


# Initializing testing and training dataset
train_dataset = ProductsDataset(csv_file="training_set.csv",
                                  root_dir="images", transform=ToTensor())

test_dataset = ProductsDataset(csv_file="test.csv",
                              root_dir="images", transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Loading model augmenting architecture for current dataset
model_conv=models.resnet50(pretrained=False)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 4)
model_conv = model_conv.to(device)

# Defining loss and optimizer
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model_conv.parameters(), lr=learning_rate)

# Training function
def train(epoch):
    model_conv.train()
    for i, sample in enumerate(train_loader):
        data, target = sample['image'].to(device), sample['coordinates'].to(device)

        optimizer.zero_grad()
        output = model_conv(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if(i%10==0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i*len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item()))

        PATH = "D:/Sahil/Dataset/resnet.pt"

    torch.save({'epoch':epoch,
                'model_state_dict':model_conv.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':loss}, PATH)


# Testing function
def test():
    with open('output.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)

        model_conv.eval()
        test_loss = 0
        correct = 0
        i=0
        for sample in test_loader:

            data, target = sample['image'].to(device=device), sample['coordinates'].to(device=device)
            output = model_conv(data)

            writer.writerow(output.detach().cpu().numpy()[0])
            writer.writerow(output.detach().cpu().numpy()[1])
            writer.writerow(output.detach().cpu().numpy()[2])
            try:
                writer.writerow(output.detach().cpu().numpy()[3])
            except:
                pass
            print(i/len(test_loader))
            i+=1


    csvFile.close()

# Training loop
for epoch in range(1, 21):
    train(epoch)

# Testing loop
with torch.no_grad():
    test()
