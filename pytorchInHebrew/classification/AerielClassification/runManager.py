from collections import namedtuple
from itertools import product
from time import time, sleep
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torch


class RunBuilder:
    """
    returns tuples (learning_rate, batch_size, shuffle...)
    """
    @staticmethod
    def get_runs(params):
        Run = namedtuple('run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


class RunManager:
    class Epoch:
        def __init__(self):
            self.loss = {'train': 0., 'val': 0.}
            self.num_correct = {'train': 0, 'val': 0}
            self.start_time = None
            self.batches = {'train': 0, 'val': 0}

    def __init__(self):
        self.run_params = None
        self.run_count = 0
        self.run_data = list()
        self.run_start_time = None
        self.network = None
        self.loader = None
        self.tb = None
        self.epoch = None
        self.epoch_count = 0
        self.phase = None

    def begin_run(self, run, network, loader):
        print(run)
        self.run_start_time = time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        # images, labels = next(iter(self.loader))
        # grid = torchvision.utils.make_grid(images)
        #
        # self.tb.add_image('images', grid)
        # self.tb.add_graph(self.network, images)

    def end_run(self):
        self.tb.close()

    def begin_epoch(self):
        self.epoch = self.Epoch()
        self.epoch.start_time = time()
        self.epoch_count += 1

    def end_epoch(self):
        epoch_duration = time() - self.epoch.start_time
        run_duration = time() - self.run_start_time

        train_loss = self.epoch.loss['train'] / len(self.loader['train'].dataset)
        train_accuracy = self.epoch.num_correct['train'] / len(self.loader['train'].dataset)
        val_loss = self.epoch.loss['val'] / len(self.loader['val'].dataset)
        val_accuracy = self.epoch.num_correct['val'] / len(self.loader['val'].dataset)
        # self.tb.add_scalar('Loss', loss, self.epoch_count)
        # self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        print('epoch: {}, epoch duration: {}, overall duration: {}'.format(self.epoch_count, epoch_duration/60, run_duration/60), flush=True)
        print('train: , loss: {}, Accuracy: {}'.format(train_loss, train_accuracy), flush=True)
        print('val: , loss: {}, Accuracy: {}'.format(val_loss, val_accuracy), flush=True)
        sleep(0.5)
        # for name, param in self.network.named_parameters():
        #     self.tb.add_histogram(name, param, self.epoch_count)
        #     self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

    def track_loss(self, loss):
        self.epoch.loss[self.phase] += (loss.item() * self.loader[self.phase].batch_size)

    @torch.no_grad()
    def track_num_correct(self, preds, labels):
        self.epoch.num_correct[self.phase] += self.get_num_correct(preds, labels)
        self.epoch.batches[self.phase] += 1

    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()

    def set_status(self, phase):
        if phase == 'train':
            self.phase = 'train'
            self.network.train()
        else:
            self.phase = 'val'
            self.network.eval()

    def save_run_model(self, run_description, model_param, optim_pram):
        state = {
            'run_param': run_description,
            'state_dict': model_param,
            'optim_state': optim_pram
        }
        torch.save(state, run_description)

    def load_run_model(self, path):
        return torch.load(path)
