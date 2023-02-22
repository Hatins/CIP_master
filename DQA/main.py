import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torchvision.models as md
import datetime
from torch.optim.lr_scheduler import MultiStepLR
from categorical_inference_poisoning.function import *
from utils.Ptloader import Ptloader


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQA',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--attack',type=str,default='IDA',choices=['KNOCKOFF','DFME','IDA'],help='DQA and IDA share the same setting')
    parser.add_argument('--dataset',type=str,default='CIFAR10',choices=['CIFAR10', 'CIFAR100', 'FOOD101'], help='choosing the dataset for victim model')
    parser.add_argument('--target_model',type=str,default='wrn28',choices=['resnet18','wrn28'],help = 'choosing the model for victim model')
    parser.add_argument('--Surrogate_model', type=str, default='wrn28', choices=['resnet18', 'wrn28'], help='choosing the model for victim model')
    parser.add_argument('--num_classes',type=int,default=10,help='setting the num_classes for victim model training')
    parser.add_argument('--size', type=int, default=32, help='setting the image size')
    parser.add_argument('--epoch',type=int,default=100,help='setting the epoch for victim model training')
    parser.add_argument('--lr',type=float,default=0.1,help='setting the learning rate for victim model training')
    parser.add_argument('--batch',type=int,default=128,help='setting the batch_size for victim model training')
    parser.add_argument('--num_workers',type=int,default=0,help='setting the num_worker for victim model training')
    parser.add_argument('--poison', type=bool, default=True, help='whether poisoning')
    parser.add_argument('--ratio',type=float,default=87,help='setting the poison ratio')
    parser.add_argument('--trigger_path',type=str,default='./trigger',help='setting the trigger path')
    parser.add_argument('--Dratio',type=float,default=1,help='Deceptive ratio')
    parser.add_argument('--method',type=str,default='DAWN',choices=['DAWN','CIP'],help = 'choose the defense method')
    args = parser.parse_args()
    device = torch.device('cuda:0')
    if args.dataset == 'CIFAR10':
        data_exp = 'CIFAR100'
    else:
        data_exp = '---------'
    if args.trigger_path != None:
        args.trigger_path = args.trigger_path + '/' + args.dataset + '_' + args.Surrogate_model + '.pt'


    if args.dataset == 'CIFAR10':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
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
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
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

    transform_totensor = transforms.Compose(
        [
            torchvision.transforms.Resize([args.size,args.size]),
            torchvision.transforms.ToTensor()
        ]
    )
    '''setting the Victim model and Surrogate model'''
    if args.target_model == 'resnet18':
        target_model = md.resnet18(num_classes=args.num_classes)
    elif args.target_model == 'wrn28':
        target_model =  WideResNet(depth=28,num_classes=args.num_classes,widen_factor=10,dropRate=0.3)

    if args.poison == False or args.method == 'DAWN':
        pretrained_file_path = '../model_train/Trained/' + args.attack
        pretrained_file_list = os.listdir(pretrained_file_path)
        for i in range(len(pretrained_file_list)):
            if (args.dataset in pretrained_file_list[i]) and (data_exp not in pretrained_file_list[i]):
                pretrained_file = pretrained_file_path + '/' + pretrained_file_list[i]
                print('loading pretrained file ', pretrained_file)
    else:
        pretrained_file_path = '../OOD_Detection/Trained/' + args.attack
        pretrained_file_list = os.listdir(pretrained_file_path)
        for i in range(len(pretrained_file_list)):
            if (args.dataset in pretrained_file_list[i]) and (data_exp not in pretrained_file_list[i]):
                pretrained_file = pretrained_file_path + '/' + pretrained_file_list[i]
                print('loading pretrained file ', pretrained_file)

    target_model.load_state_dict(torch.load(pretrained_file, map_location='cuda:0'))
    target_model = target_model.to(device)

    if args.Surrogate_model == 'resnet18':
        Surrogate_model = md.resnet18(num_classes=args.num_classes)
    elif args.Surrogate_model == 'wrn28':
        Surrogate_model =  WideResNet(depth=28,num_classes=args.num_classes,widen_factor=10,dropRate=0.3)

    '''setting the test set'''
    test_path = r'../model_train/Dataset/' + '/' + args.dataset + '/' + args.dataset + '_all' + '/test'
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform_test)


    print('setting the transfer set')
    query_set_path = '../IDA/dataset/'+ args.dataset + '_half'
    filelist = os.listdir(query_set_path)  
    target_model.eval()
    query_times = len(filelist)
    args.query_times = query_times

    detecitve_times = int(args.query_times/args.Dratio)
    detecitve_file_path = './Noise'
    detecitve_file = os.listdir(detecitve_file_path)

    loop_save = tqdm(enumerate(filelist), total=args.query_times,position=0)
    loop_save.set_description(f'transfer set establishing...')

    loop_detective = tqdm(enumerate(detecitve_file),total=detecitve_times,position=0)
    loop_detective.set_description('detective set establishing...')

    image_list = []
    probability_list = []

    for index, data in loop_save:
        if query_times > 0:
            image_name = query_set_path + '/' + data
            image_PIL = Image.open(image_name).convert('RGB')
            image_save = transform_train(image_PIL)
            image_list.append(image_save)
            query_times = query_times - 1

    for index, data in loop_detective:
        if detecitve_times > 0:
            image_name = detecitve_file_path + '/' + data
            image_PIL = Image.open(image_name).convert('RGB')
            image_save = transform_train(image_PIL)
            image_list.append(image_save)
            detecitve_times = detecitve_times - 1

    image_tensor = torch.stack([i for i in image_list], dim=0)

    if args.method == 'CIP':
        output_list = poison_fun_tensor(target_model,image_tensor,None,args.attack,args.ratio,args.dataset,args.trigger_path,upside_down,poison = args.poison)
    elif args.method == 'DAWN':
        output_list = DAWN_poison(target_model, image_tensor, None, args.attack, args.ratio, args.dataset,args.trigger_path,poison=args.poison)

    output_tensor = torch.stack([i for i in output_list],dim=0)

    image_train_tensor = image_tensor[:args.query_times]
    output_train_tensor = output_tensor[:args.query_times]

    dataset = TensorDataset(image_train_tensor, output_train_tensor)
    train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    if args.poison == True:
        trigger_path = './trigger' + '/' + args.dataset + '_' + args.Surrogate_model + '.pt'
        trigger_set = Ptloader(trigger_path, train=True)
        trigger_loader = DataLoader(trigger_set, batch_size=args.batch, shuffle=False)

    '''training surrogate model'''

    Surrogate_model = Surrogate_model.to(device)
    # optimizer = optim.Adam(Surrogate_model.parameters(), lr=args.lr)
    optimizer = optim.SGD(Surrogate_model.parameters(), lr=args.lr,momentum=0.9,nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
    accuracy_test_list = []
    accuracy_best = 0
    for each_epoch in range(args.epoch):
        Surrogate_model.train()
        Surrogate_model = Surrogate_model.to(device)
        criterion = nn.CrossEntropyLoss()

        start_time = datetime.datetime.now()
        correct_train_number = 0
        loop_train = tqdm(enumerate(train_loader), total=len(train_loader),position=0)
        if args.poison == True:
            loop_trigger = tqdm(trigger_loader, total=len(trigger_loader), position=0)
        loop_test = tqdm(test_loader, total=len(test_loader), position=0)
        for index, (inputs, probability) in loop_train:
            inputs, probability = inputs.to(device), probability.to(device)
            outputs = Surrogate_model(inputs)
            optimizer.zero_grad()
            loss = soft_cross_entropy(outputs, probability)
            loss.backward()
            optimizer.step()
            pred = outputs.argmax(dim=1)
            labels_item = probability.argmax(dim=1)
            correct_train_number += pred.eq(labels_item.view_as(pred).to(device)).sum().item()
            accuracy_train_show = correct_train_number / args.query_times
            loop_train.set_description(f'training_Epoch [{each_epoch + 1}/{args.epoch}]')
            loop_train.set_postfix(loss=loss.item(), acc_train=accuracy_train_show)

        scheduler.step()
        correct_test_number = 0

        Surrogate_model.eval()
        if args.poison == True:
            correct_trigger_number = 0
            with torch.no_grad():
                for data, target in loop_trigger:
                    data, target = data.to(device), target.to(device)
                    outputs = Surrogate_model(data)
                    pred = outputs.argmax(dim=1)
                    target = target.argmax(dim=1)
                    correct_trigger_number += pred.eq(target.view_as(pred)).sum().item()
                    accuracy_trigger_show = correct_trigger_number / len(trigger_set)
                    loop_trigger.set_description(f'trigger__Validation')
                    loop_trigger.set_postfix(acc_trigger=accuracy_trigger_show)
                accuracy_trigger = correct_trigger_number / len(trigger_set)

        with torch.no_grad():
            for data, target in loop_test:
                data, target = data.to(device), target.to(device)
                outputs = Surrogate_model(data)
                pred = outputs.argmax(dim=1)
                correct_test_number += pred.eq(target.view_as(pred)).sum().item()
                accuracy_test_show = correct_test_number / len(test_set)
                loop_test.set_description(f'testing__Epoch [{each_epoch + 1}/{args.epoch}]')
                loop_test.set_postfix(acc_test=accuracy_test_show)
            accuracy_test = correct_test_number / len(test_set)
            accuracy_test_list.append(accuracy_test * 100)

        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
        print('the best accuracy is {}'.format(accuracy_best,'.2f'))























