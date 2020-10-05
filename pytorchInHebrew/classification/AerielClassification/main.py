import PIL
import torch
import torch.nn as nn
from runManager import RunBuilder, RunManager
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms, models
from collections import OrderedDict
from CustonDataset import CustomDataset, SubSet
import torch.nn.functional as F
import utills
from tqdm import tqdm
from math import floor, ceil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        # num of kernels = out_channels
        # formula = Out = (W−F+2P) / S + 1
        # n = 650
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        # Out = (n − 3) + 1 = 206
        # 204
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        # 256
        # Out = (n − 3) + 1 = 128
        # an affine operation: y = Wx + b
        # linear/dense/fully connected layer are the same thing
        # it calls features because its a one dimension tensor (after flatten)
        # out_features X in_features * in_features X 1 = out_features X 1 which equals to the vector of the next layer

        # in_features =
        self.fc1 = nn.Linear(in_features=16 * 50 * 50, out_features=500, bias=True)  # 11*11 from image dimension
        self.fc2 = nn.Linear(in_features=500, out_features=200, bias=True)
        self.fc3 = nn.Linear(in_features=200, out_features=18, bias=True)

    """
    forword function is part of the api of every layer and network.
    it gets a tensor, and pass it through every layer.
    this is really the transformation.
    it is lounch by the call method
    """

    def forward(self, t):
        # Max pooling over a (2, 2) window
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # If the size is a square you can only specify a single number
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # flatten only the images tensors themselves and not the whole input_tensor
        t = t.view(-1, self.num_flat_features(t))
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        out_tensor = self.fc3(t)
        return out_tensor

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


csv_train_dir = r"C:\Users\lior\PycharmProjects\untitled1\training\data\images\AID/train_csv.csv"
csv_test_dir = r"C:\Users\lior\PycharmProjects\untitled1\training\data\images\test"
main_root = r"C:\Users\lior\PycharmProjects\untitled1\training\data\images\AID"

IMG_SIZE = 224
dataset = CustomDataset(csv_file=csv_train_dir, root_dir=main_root)
train_subset, val_subset = torch.utils.data.random_split(dataset, [floor(len(dataset) * 0.9), ceil(len(dataset) * 0.1)])

train_set = SubSet(train_subset)
val_set = SubSet(val_subset)

hyp_parameters = OrderedDict(
    epochs_num=[2],
    lr=[0.001],
    batch_size=[64],
    shuffle=[True],
)

mean = torch.tensor(291.)
std = torch.tensor([348.8725])

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage("RGB"),
        transforms.Resize((IMG_SIZE, IMG_SIZE), PIL.Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.CenterCrop(size=IMG_SIZE),
        transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(*utills.mean_std(train_set, (mean, std)))
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage("RGB"),
        transforms.Resize((IMG_SIZE, IMG_SIZE), PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(*utills.mean_std(val_set, (mean, std)))
    ])
}

train_set.transform = data_transforms["train"]
val_set.transform = data_transforms["val"]

m = RunManager()

data_loader = {'train': None, 'val': None}


def get_model_optimizer(run, from_scratch):
    if from_scratch:
        # net = torch.load("saved_model_batch=20_lr=0.01_shuff=True")
        net = models.wide_resnet101_2(pretrained=True)
        n_act = net.fc.in_features
        for param in net.parameters():
            param.requires_grad = False
        net.fc = nn.Sequential(nn.Linear(n_act, 512), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(512, 17))
        optimizer = Adam(net.parameters(), lr=run.lr)
    else:
        state = m.load_run_model("saved_model_batch=64_lr=0.001_shuff=True.pt")
        net = models.wide_resnet101_2(pretrained=False)
        n_act = net.fc.in_features
        for param in net.parameters():
            param.requires_grad = False
        net.fc = nn.Sequential(nn.Linear(n_act, 512), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(512, 17))
        net.load_state_dict(state['state_dict'])
        optimizer = Adam(net.parameters(), lr=run.lr)
        optimizer.load_state_dict(state['optim_state'])
    return net, optimizer


if __name__ == '__main__':
    for run in RunBuilder.get_runs(hyp_parameters):
        data_loader['train'] = DataLoader(dataset=train_set, batch_size=run.batch_size, shuffle=run.shuffle)
        data_loader['val'] = DataLoader(dataset=val_set, batch_size=run.batch_size, shuffle=False)
        print("-- finished loading data --")
        net, optimizer = get_model_optimizer(run, False)
        scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.2)
        m.begin_run(run, net, data_loader)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(run.epochs_num):
            m.begin_epoch()
            for phase in ["train", "val"]:
                m.set_status(phase)
                for images, labels in tqdm(data_loader[phase], desc=phase):
                    images = images.to(device)
                    labels = labels.to(device)
                    # eval without dropout
                    with torch.set_grad_enabled(phase == 'train'):
                        preds = net(images)
                        loss = loss_fn(preds, labels.argmax(dim=1))
                        optimizer.zero_grad()
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    m.track_loss(loss)
                    m.track_num_correct(preds, labels)
            m.end_epoch()
            scheduler.step()
        m.save_run_model(
            "saved_model_batch={}_lr={}_shuff={}.pt".format(run.batch_size, run.lr, run.shuffle),
            net.state_dict(),
            optimizer.state_dict()
        )
    m.end_run()
