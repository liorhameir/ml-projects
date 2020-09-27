# import numpy as np
# import cv2
# import torch
# import torchvision
# from math import floor, ceil
#
# im = cv2.imread(r'C:\Users\lior\PycharmProjects\untitled1\training\data\images\AID\Untitled.jpg')
#
# # kernel = np.array([[-1, 1, 0],
# #                    [1, 4, 0],
# #                    [-1, 1, 0]])
#
# kernel = np.array([[0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0],
#                    [0, 1, 2, 1, 0],
#                    [-0.2, -1, -1, -1, -0.2],
#                    [0, 0, 0, 0, 0]])
#
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#
# gray = gray/255.0
# image = im / 255.0
# filtered = cv2.filter2D(src=image, kernel=kernel, ddepth=-1)
# grey_filtered = cv2.filter2D(src=gray, kernel=kernel, ddepth=-1)
# filtered = cv2.resize(filtered, (800, 660))
# grey_filtered = cv2.resize(grey_filtered, (800, 660))
# image = cv2.resize(image, (800, 660))
#
# filterd_img = np.concatenate((filtered, image), axis=1)
# grey_filtered_img = np.concatenate((grey_filtered, image), axis=1)
#
# cv2.imshow('Numpy Concat', filterd_img)
# # cv2.imshow('Numpy Horizontal Concat', grey_filtered_img)
#
# cv2.waitKey()
# cv2.destroyAllWindows()


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time, sleep


t = torch.tensor([1, 2, 3], dtype=torch.int8)
t = t / 0.3


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_set = torchvision.datasets.MNIST('/files/',
                                       train=True,
                                       download=True,
                                       transform=transform)


train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=64,
    shuffle=True)


images, labels = next(iter(train_loader))
print("the shape is ", labels.shape)
print(labels)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(images[i][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])

fig.show()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=20,
                               kernel_size=5,
                               stride=1)

        self.conv2 = nn.Conv2d(in_channels=20,
                               out_channels=50,
                               kernel_size=5,
                               stride=1)

        self.fc1 = nn.Linear(in_features=50 * 4 * 4,
                             out_features=500)
        self.fc2 = nn.Linear(in_features=500,
                             out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = t.view(t.size(0), -1)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        return t


net = Net()
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)


def train(net, epoch_num):
    start = time()
    for epoch in range(1, epoch_num + 1):
        running_loss = 0.
        correct = 0.
        msg = 'training: epoch {}/{}'.format(epoch, epoch_num + 1)
        for images, labels in tqdm(train_loader, desc=msg):
            images = images.to(device)
            labels = labels.to(device)
            print(images)
            # נאפס את הגרדיאנט
            optimizer.zero_grad()
            # נזין לרשת את הbatch
            preds = net(images)
            # נחשב loss
            loss = criterion(preds, labels)
            # נחשב gradients
            loss.backward()
            # האופטימייזר יתקן את המשקלים והbiases
            optimizer.step()
            # נוסיף את הטעות של הbatch לטעות הכללית של כל הepoch
            running_loss += loss.item()
            # נחשב כמה תמנוות בbatch חזינו נכון
            correct += preds.argmax(dim=1).eq(labels).sum().item()

        print('results: Epoch {}, accuracy {}, loss: {}\n'.format(
            epoch,
            (correct / len(train_loader.dataset)) * 100,
            (running_loss / len(train_loader))), flush=True)

        sleep(.5)

    end = time()
    print('Done Training')
    print('%0.2f minutes' % ((end - start) / 60))


net.train()
train(net, 10)

test_set = torchvision.datasets.MNIST('/files/',
                                      train=True,
                                      download=True,
                                      transform=transform)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=64,
    shuffle=True)

correct = 0
net.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = net(images)
        correct += preds.argmax(dim=1).eq(labels).sum().item()

print('Accuracy of on test images: {}'.format(
        100 * correct / len(test_loader.dataset)))
