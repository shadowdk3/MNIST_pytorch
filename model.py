import torch
import torch.nn as nn
import torch.nn.functional as F

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output

# # building a linear stack of layers with the sequential model
# model = Sequential()
# # convolutional layer
# model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(28,28,1)))
# model.add(MaxPool2D(pool_size=(1,1)))
# # flatten output of conv
# model.add(Flatten())
# # hidden layer
# model.add(Dense(100, activation='relu'))
# # output layer
# model.add(Dense(10, activation='softmax'))

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(1)
        self.fc1 = nn.Linear(25*26*26, 100)
        self.fc2 = nn.Linear(100, 10)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x) # [batch_size, 25, 26, 26]
        x = x.view(x.size(0), -1)   # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

