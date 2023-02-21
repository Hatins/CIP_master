import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import torchvision.models as models
from model.wrn import WideResNet

from utils.Cutout import Cutout

import argparse
device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description='Training the target (victim) model before surrogate attack', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',type=str, default='CIFAR10', choices=['MNIST', 'Fashion_MNIST', 'CIFAR10', 'CIFAR100'], help='dataset selection')
parser.add_argument('--model',type=str,default='resnet18', choices=['resnet18','wrn_28','vgg16'], help = 'victim (to-be-protected) model')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--partition',type=str, default='all', choices=['all', 'half'], help = 'data partition')
parser.add_argument('--batch',type=int, default=128, help='batchsize')
parser.add_argument('--num_worker', type=int, default=4, help='number of workers')
args = parser.parse_args()


def train(model, train_set, train_loader, test_set, test_loader, partition='all'):
    print('Training model...')
    num_params = sum(i.numel() for i in model.parameters() if i.requires_grad)
    print('Number of parameters: {}'.format(num_params))
    accuracy_test_list = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    for epoch in range(args.epoch):
        #training
        model = model.to(device)
        model.train()
        correct_train_number = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        #testing
        correct_test_number = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                correct_test_number += pred.eq(labels.view_as(pred)).sum().item()
            test_acc_per_epoch = correct_test_number / len(test_set)
        save_name  = './trained/' + args.partition + '/' + args.dataset+'_'+args.model+'_epoch_{}_test_accuracy_{:.2f}%.pt'.format(epoch, test_acc_per_epoch * 100)
        accuracy_test_list.append(test_acc_per_epoch * 100)
        torch.save(model.state_dict(), save_name)

        print('Epoch {}: test accuracy {:.2f}%, best test accuracy {:.2f}%'
              .format(epoch, test_acc_per_epoch * 100, np.array(accuracy_test_list).max()))

    accuracy_test_list = np.array(accuracy_test_list)
    save_best_name = './trained/' + args.partition + '/' + args.dataset+'_'+args.model+'_epoch_{}_test_accuracy_{:.2f}%.pt'.format(accuracy_test_list.argmax(), accuracy_test_list.max())
    
    save_path = os.listdir('./trained/' + args.partition)
    for file_name in save_path:
        file_name_need_remove = r'./trained/' + args.partition + '/' + file_name
        if file_name_need_remove != save_best_name and args.dataset in file_name_need_remove:
            os.remove(file_name_need_remove)

if __name__ == '__main__':

    '''data preparation'''
    if args.dataset == 'MNIST':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif args.dataset == 'Fashion_MNIST':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif args.dataset == 'CIFAR10':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            Cutout(n_holes=1,length=16)
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    elif args.dataset == 'CIFAR100':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            Cutout(n_holes=1, length=16)
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    '''model preparation'''
    if args.model == 'resnet18':
        model = models.resnet18(num_classes=args.num_classes)
    elif args.model == 'wrn_28':
        model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10, dropRate=0.3)
    elif args.model == 'vgg16':
        model = models.vgg16_bn(num_classes=args.num_classes)

    train_path = r'./Dataset' + '/'+ args.dataset + '/'+args.dataset+'_'+args.partition+'/train'
    test_path = r'./Dataset' + '/' + args.dataset + '/'+args.dataset+'_'+args.partition+'/test'

    train_set = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_worker)


    train(model, train_set, train_loader, test_set, test_loader, args.partition)

    print('End.')