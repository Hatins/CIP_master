import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader
import numpy as np
import os
import datetime
from tqdm import tqdm
from Model.wrn import WideResNet
from Utils.Cutout import Cutout
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.models as md
import argparse
device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description='Training the Victim model for surrogate attack',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--attack',type=str,default='Datafree',choices=['Knockoff','Datafree','IDA'],help='choosing the attack forms')
parser.add_argument('--dataset',type=str,default='Mnist',choices=['Mnist','Gtsrb','Cifar10','Cifar100','Fashion_mnist','FOOD101'], help='choosing the dataset for victim model')
parser.add_argument('--model',type=str,default='resnet18',choices=['resnet18','wrn_28','vgg16'],help = 'choosing the model for victim model')
parser.add_argument('--num_classes',type=int,default=10,help='setting the num_classes for victim model training')
parser.add_argument('--epoch',type=int,default=100,help='setting the epoch for victim model training')
parser.add_argument('--lr',type=float,default=0.1,help='setting the learning rate for victim model training')
parser.add_argument('--type',type=str,default='all',choices=['all','half'],help = 'whether the half or full')
parser.add_argument('--batch',type=int,default=128,help='setting the batch_size for victim model training')
parser.add_argument('--num_worker',type=int,default=4,help='setting the num_worker for victim model training')
args = parser.parse_args()


# shutil.rmtree('./Trained/'+args.attack+'/Tensorboard')
# os.mkdir('./Trained/'+args.attack+'/Tensorboard')
writer = SummaryWriter('./Trained/'+args.attack+'/Tensorboard')


def train(model,train_set, train_loader, test_set, test_loader):
    print('Victim model is training')
    num_params = sum(i.numel() for i in model.parameters() if i.requires_grad)
    print('the number of parameters is {}'.format(num_params))
    Victim_model = model.to(device)
    accuracy_test_list = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Victim_model.parameters(), lr=args.lr,momentum=0.9,nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    start_time = datetime.datetime.now()

    for each_epoch in range(args.epoch):
        Victim_model = Victim_model.to(device)
        Victim_model.train()

        #training
        correct_train_number = 0
        loop_train = tqdm(enumerate(train_loader), total=len(train_loader))
        for index, (inputs, labels) in loop_train:
            inputs, lables = inputs.to(device), labels.to(device)
            outputs = Victim_model(inputs)
            pred = outputs.argmax(dim=1)
            correct_train_number += pred.eq(labels.view_as(pred).to(device)).sum().item()
            loss = criterion(outputs, lables)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy_train_show = correct_train_number / len(train_set)
            loop_train.set_description(f'training_Epoch [{each_epoch + 1}/{args.epoch}]')
            loop_train.set_postfix(loss=loss.item(), acc_train=accuracy_train_show)
        writer.add_scalar('train_loss',loss.item(),each_epoch)
        scheduler.step()

        #testing
        loop_test = tqdm(test_loader, total=len(test_loader))
        correct_test_number = 0
        Victim_model.eval()
        with torch.no_grad():
            for inputs, labels in loop_test:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = Victim_model(inputs)
                pred = outputs.argmax(dim=1)
                correct_test_number += pred.eq(labels.view_as(pred)).sum().item()
                accuracy_test_show = correct_test_number / len(test_set)
                loop_test.set_description(f'testing__Epoch [{each_epoch + 1}/{args.epoch}]')
                loop_test.set_postfix(acc_test=accuracy_test_show)
            accuracy_test = correct_test_number / len(test_set)
            writer.add_scalar('test_acc',accuracy_test,each_epoch)
        save_name = './Trained/' + args.attack + '/' + args.dataset+'_'+args.model+'_epoch_{}_accuracy_{:.2f}%.pt'.format(
            each_epoch, accuracy_test * 100)
        accuracy_test_list.append(accuracy_test * 100)
        torch.save(Victim_model.state_dict(), save_name)
    endtime = datetime.datetime.now()
    accuracy_test_list = np.array(accuracy_test_list)
    print('the best accurary is {:.2f}'.format(accuracy_test_list.max()))
    save_best_name = './Trained/' +  args.attack + '/' +  args.dataset+'_'+args.model+'_epoch_{}_accuracy_{:.2f}%.pt'.format(
        accuracy_test_list.argmax(), accuracy_test_list.max())
    save_path = os.listdir(r'./Trained/' + args.attack)
    for file_name in save_path:
        file_name_need_remove = r'./Trained/' + args.attack + '/' + file_name
        Tensorboard_file_path = r'./Trained/' + args.attack + '/' + 'Tensorboard'
        if file_name_need_remove != save_best_name and file_name_need_remove != Tensorboard_file_path and args.dataset in file_name_need_remove:
            os.remove(file_name_need_remove)
    print('training time is {}s'.format((endtime - start_time).seconds))

if __name__ == '__main__':
    '''setting the transform'''
    if args.dataset == 'Mnist':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        # transform_train.transforms.append(Cutout(n_holes=1, length=16))
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif args.dataset == 'Cifar10':
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
    elif args.dataset == 'Cifar100':
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
    elif args.dataset == 'Gtsrb':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean= np.array([0.3337, 0.3064, 0.3171]), std= np.array([0.2672, 0.2564, 0.2629]) )]
        )
        transform_test = transform_train
    elif args.dataset == 'Fashion_mnist':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        # transform_train.transforms.append(Cutout(n_holes=1, length=16))
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif args.dataset == 'FOOD101':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5567, 0.4381, 0.3198), (0.2591, 0.2623, 0.2633))
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5567, 0.4381, 0.3198), (0.2591, 0.2623, 0.2633))
        ])

    '''setting the model'''
    if args.model == 'resnet18':
        Victim_model = md.resnet18(num_classes=args.num_classes)
    elif args.model == 'wrn_28':
        Victim_model = WideResNet(depth=28,num_classes=args.num_classes,widen_factor=10,dropRate=0.3)
    elif args.model == 'vgg16':
        Victim_model = md.vgg16_bn(num_classes=args.num_classes)

    train_path = r'./Dataset/' + '/'+ args.dataset + '/'+args.dataset+'_'+args.type+'/train'
    test_path = r'./Dataset/' + '/' + args.dataset + '/'+args.dataset+'_'+args.type+'/test'

    train_set = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_worker)


    train(Victim_model, train_set, train_loader, test_set, test_loader)