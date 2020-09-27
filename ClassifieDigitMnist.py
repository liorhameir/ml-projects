import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time, sleep

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

criterion = nn.CrossEntropyLoss()

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
