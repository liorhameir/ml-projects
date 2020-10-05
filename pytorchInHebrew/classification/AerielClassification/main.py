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

csv_dir = r"PATH_TO_CSV/train_csv.csv"
main_root = r"PATH_TO_DIR\AID"

IMG_SIZE = 224
dataset = CustomDataset(csv_file=csv_dir, root_dir=main_root)
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
