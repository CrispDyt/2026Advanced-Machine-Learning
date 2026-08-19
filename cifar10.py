#!/usr/bin/python

import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# Load the CIFAR10 data
# This code downloads the data, unless you already have it.
# The output of torchvision datasets are PILImage images of range [0, 1].
# The transform normalises the images to have mean 0 and sd 1.

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223,  0.24348513,  0.26158784))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                        download=True, transform=transform)
# batch_size =4 means we only randomly select 4 images from the training set to train the network.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# load test data
testset = torchvision.datasets.CIFAR10(root='~/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# define images classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


########################################################################
# Examine images

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 4 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# Define a Convolution Neural Network
# The images are three channel (R, G, B). 
# for each channel, the size could be 60*60 
# 60*60*3  

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 inputs, 6 outputs, kernels are size 5x5
        # provides 18 kernels
        # https://pytorch.org/docs/stable/nn.html#conv2d
        self.conv1 = nn.Conv2d(3, 6, 5) # CNN layer 
        # Pool 2x2 sets of pixels with stride size 2
        # https://pytorch.org/docs/stable/nn.html#maxpool2d
        self.pool = nn.MaxPool2d(2, 2)
        # 6 inputs, 16 outputs, with kernel size 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 8, 3)
        
        # https://pytorch.org/docs/stable/nn.html?highlight=eval#linear
        self.fc1 = nn.Linear(8 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        # final layer outputs 10 classes
        self.fc3 = nn.Linear(84, 10)
        self.do = nn.Dropout(p=0.5) 
        self.pad = nn.ZeroPad2d(1)

    def forward(self, x):
        # x begins as a 32x32 pixel, 3 channel (R,G,B)image
        x = self.pool(F.relu(self.conv1(x)))
        # after convolution with 5x5 kernel, image is 28x28, then pooled to 14x14
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pad(x)
        # after convolution with 5x5 kernel, image is 10x10, then pooled to 5x5
        # giving 16 outputs of size 5x5.y
        x = F.relu(self.conv3(x))
        x = x.view(-1, 8 * 5 * 5)
        # flatten 16x5x5 tensor to 16x5x5=400 length vector
        x = self.do(x)

        # 400 dimensions to 120
        x = F.relu(self.fc1(x))
        # 120 dimensions to 84
        x = F.relu(self.fc2(x))
        # 84 dimensions to 10
        x = self.fc3(x)
        return x


# Create the network
net = Net()
net.train() # set the network to training mode


########################################################################
# Define a Loss function and optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    # iterate through batches
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # forward pass
        loss = criterion(outputs, labels) # compute loss
        loss.backward() # backwards pass
        optimizer.step() # compute gradient and update weights

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

########################################################################
# 5. Test the network on the test data

net.eval() # set the network to evaluation mode

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# predict the class probability of the test images
outputs = net(images)

# the class of the image is associated to the greatest probability
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# Now compute the accuracy for all test images.
correct = 0
total = 0
# with torch.no_grad() means we frozen theses parameters 
with torch.no_grad(): 
    # for each batch
    for data in testloader:
        images, labels = data
        outputs = net(images) # predict the probability of each class
        _, predicted = torch.max(outputs.data, 1) # choose the class with max probability
        total += labels.size(0) # add to total count
        correct += (predicted == labels).sum().item() # add to correct count

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# Compute the accuracy of each class

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))




# View the trained kernels, do they contain shapes from the pictures?
weights1 = net.conv1.weight.data.numpy()
weights2 = net.conv2.weight.data.numpy()
# show the first kernel
plt.imshow(weights1[0, 0, :, :])
