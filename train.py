import time
import argparse
import numpy as np
import torch.optim as optim

import torch
import torch.nn as nn
import os

from model import GGNN
from utils.data.datadetect import DataDetect
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--state_dim', type=int, default=50, help='GGNN hidden state size')
parser.add_argument('--graphstate_dim', type=int, default=15, help='GGNN state vector size')
parser.add_argument('--hidden_dim', type=int, default=15, help='GGNN state vector output hidden state size')

parser.add_argument('--n_steps', type=int, default=10, help='propogation steps number of GGNN')

train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='train',
                    type=str, help='train or test')
parser.add_argument('--dataset_root', default='./dataset',
                    help='Dataset root directory path')
parser.add_argument('--num_epoch', default=500, type=int,
                    help='Num epoch for training')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--num_worker', default=12, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--num_classes', default=2, type=int,
                    help='Number of class used in model')
parser.add_argument('--device', default=[0], type=list,
                    help='Use CUDA to train model')
parser.add_argument('--grad_accumulation_steps', default=1, type=int,
                    help='Number of gradient accumulation steps')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder', default='./saved/weights', type=str,
                    help='Directory for saving checkpoint models')
args = parser.parse_args()
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


def prepare_device(device):
    n_gpu_use = len(device)
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    list_ids = device
    device = torch.device('cuda:{}'.format(device[0]) if n_gpu_use > 0 else 'cpu')

    return device, list_ids


def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict


dataset = DataDetect(os.path.join(args.dataset_root, args.dataset))
args.num_nodes = dataset.max_num_nodes
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_worker,
                        shuffle=True,
                        )

model = GGNN(opt=args)
device, device_ids = prepare_device(args.device)
model = model.to(device)
if (len(device_ids) > 1):
    model = torch.nn.DataParallel(model, device_ids=device_ids)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
criterion = nn.CrossEntropyLoss()


def train():
    model.train()
    iteration = 1
    for epoch in range(args.num_epoch):
        print("{} epoch: \t start training....".format(epoch))
        start = time.time()
        result = {}
        total_loss = []
        optimizer.zero_grad()
        for idx, (adj_matrix, annotation, target) in enumerate(dataloader):
            adj_matrix = adj_matrix.to(device)
            annotation = annotation.to(device)
            padding = torch.zeros(len(annotation), args.n_node, args.state_dim - args.annotation_dim)
            init_input = torch.cat((annotation, padding), 2).to(device)
            classification = model(init_input, annotation, adj_matrix)
            classification_loss = criterion(classification, target)
            classification_loss = classification_loss.mean()
            loss = classification_loss
            if bool(loss == 0):
                print('loss equal zero(0)')
                continue
            loss.backward()
            if (idx + 1) % args.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()

            total_loss.append(loss.item())
            if iteration % 100 == 0:
                print('{} iteration: training ...'.format(iteration))
                ans = {
                    'epoch': epoch,
                    'iteration': iteration,
                    'cls_loss': classification_loss.item(),
                    'mean_loss': np.mean(total_loss)
                }
                for key, value in ans.items():
                    print('    {:15s}: {}'.format(str(key), value))

            iteration += 1
        scheduler.step(np.mean(total_loss))
        result = {
            'time': time.time() - start,
            'loss': np.mean(total_loss)
        }
        for key, value in result.items():
            print('    {:15s}: {}'.format(str(key), value))
        arch = type(model).__name__
        state = {
            'arch': arch,
            'num_class': args.num_classes,
            'network': args.network,
            'state_dict': get_state_dict(model)
        }
        torch.save(state, './weights/checkpoint_{}_{}.pth'.format(args.network, epoch))
    state = {
        'arch': arch,
        'options': args,
        'state_dict': get_state_dict(model)
    }
    torch.save(state, './weights/Final_{}.pth'.format(args.network))


if __name__ == '__main__':
    train()
