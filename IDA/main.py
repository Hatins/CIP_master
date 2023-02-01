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
from Adaptive_Poisoning.function import *
from Utils.Ptloader import Ptloader
from tensorboardX import SummaryWriter

writer = SummaryWriter('./Tensorboard')

def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDA',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--attack',type=str,default='IDA',choices=['Knockoff','Datafree','IDA'],help='choosing the attack forms')
    parser.add_argument('--dataset',type=str,default='Cifar10',choices=['Mnist','Cifar10','Fashion_mnist','Cifar100','FOOD101'], help='choosing the dataset for victim model')
    parser.add_argument('--Victim_model',type=str,default='wrn28',choices=['resnet18','wrn28'],help = 'choosing the model for victim model')
    parser.add_argument('--Surrogate_model', type=str, default='wrn28', choices=['resnet18', 'wrn28'], help='choosing the model for victim model')
    parser.add_argument('--num_classes',type=int,default=10,help='setting the num_classes for victim model training')
    parser.add_argument('--size', type=int, default=32, help='setting the image size')
    parser.add_argument('--epoch',type=int,default=100,help='setting the epoch for victim model training')
    parser.add_argument('--lr',type=float,default=0.1,help='setting the learning rate for victim model training')
    parser.add_argument('--batch',type=int,default=128,help='setting the batch_size for victim model training')
    parser.add_argument('--num_workers',type=int,default=4,help='setting the num_worker for victim model training')
    parser.add_argument('--poison', type=bool, default=False, help='whether poisoning')
    parser.add_argument('--ratio',type=float,default=87,help='setting the poison ratio')
    parser.add_argument('--trigger_path',type=str,default='./trigger',help='setting the trigger path')
    parser.add_argument('--method',type=str,default='DP',choices=['DAWN','CIP','DP'],help = 'choose the defense method')
    args = parser.parse_args()
    device = torch.device('cuda:0')
    if args.dataset == 'Cifar10':
        data_exp = 'Cifar100'
    else:
        data_exp = '---------'
    if args.trigger_path != None:
        args.trigger_path = args.trigger_path + '/' + args.dataset + '_' + args.Surrogate_model + '.pt'

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
            torchvision.transforms.Normalize(mean=np.array([0.3337, 0.3064, 0.3171]),
                                             std=np.array([0.2672, 0.2564, 0.2629]))]
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

    transform_totensor = transforms.Compose(
        [
            torchvision.transforms.Resize([args.size,args.size]),
            torchvision.transforms.ToTensor()
        ]
    )
    '''setting the Victim model and Surrogate model'''
    if args.Victim_model == 'resnet18':
        Victim_model = md.resnet18(num_classes=args.num_classes)
    elif args.Victim_model == 'wrn28':
        Victim_model =  WideResNet(depth=28,num_classes=args.num_classes,widen_factor=10,dropRate=0.3)

    if args.poison == False or args.method == 'DAWN' or args.method == 'DP':
        pretrained_file_path = '../Victim_Model_Train/Trained/' + args.attack
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

    Victim_model.load_state_dict(torch.load(pretrained_file, map_location='cuda:0'))
    Victim_model = Victim_model.to(device)

    if args.Surrogate_model == 'resnet18':
        Surrogate_model = md.resnet18(num_classes=args.num_classes)
    elif args.Surrogate_model == 'wrn28':
        Surrogate_model =  WideResNet(depth=28,num_classes=args.num_classes,widen_factor=10,dropRate=0.3)

    '''setting the test set'''
    test_path = r'../Victim_Model_Train/Dataset/' + '/' + args.dataset + '/' + args.dataset + '_all' + '/test'
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform_test)


    print('setting the transfer set')
    query_set_path = './dataset/'+ args.dataset + '_half'
    filelist = os.listdir(query_set_path)  # 打开对应的文件夹
    Victim_model.eval()
    query_times = len(filelist)
    args.query_times = query_times
    loop_save = tqdm(enumerate(filelist), total=args.query_times)
    loop_save.set_description(f'transfer set establishing...')
    image_list = []
    probability_list = []

    for index, data in loop_save:
        if query_times > 0:
            image_name = query_set_path + '/' + str(index+1)+'.png'  # 拼接图像的读取地址r'\{}\\'.format(item)
            image_PIL = Image.open(image_name).convert('RGB')
            image_save = transform_train(image_PIL)
            image_list.append(image_save)
            query_times = query_times - 1

    image_tensor = torch.stack([i for i in image_list], dim=0)
    if args.method == 'CIP':
        output_list = poison_fun_tensor(args,Victim_model,image_tensor,None,args.attack,args.ratio,args.dataset,args.trigger_path,upside_down,poison = args.poison)  # TODO:
    elif args.method == 'DAWN':
        output_list = DAWN_poison(Victim_model, image_tensor, None, args.attack, args.ratio, args.dataset,args.trigger_path,poison=args.poison)  # TODO:
    elif args.method == 'DP':
        output_list = DP_poison(Victim_model,image_tensor,None,poison=args.poison)
    args.poison = False

    output_tensor = torch.stack([i for i in output_list],dim=0)
    Dataset = TensorDataset(image_tensor, output_tensor)
    train_loader = DataLoader(Dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
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
        # optimizer = optim.SGD(Surrogate_model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        # optimizer = optim.Adam(Surrogate_model.parameters(), lr=args.lr)
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
            # ----------------hard label
            probability = torch.argmax(probability, dim=1)
            # probability = F.one_hot(probability)
            loss = F.cross_entropy(outputs, probability)
            # ----------------soft label
            # loss = soft_cross_entropy(outputs, probability)
            loss.backward()
            optimizer.step()
            pred = outputs.argmax(dim=1)
            # ----------------soft label
            # labels_item = probability.argmax(dim=1)
            # ----------------hard label
            labels_item = probability
            correct_train_number += pred.eq(labels_item.view_as(pred).to(device)).sum().item()
            accuracy_train_show = correct_train_number / args.query_times
            loop_train.set_description(f'training_Epoch [{each_epoch + 1}/{args.epoch}]')
            loop_train.set_postfix(loss=loss.item(), acc_train=accuracy_train_show)
        writer.add_scalar('acc_train',accuracy_train_show,each_epoch)

        scheduler.step()
        # 测试集、触发集测试
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
                writer.add_scalar('acc_tri', accuracy_trigger, each_epoch)
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
            writer.add_scalar('acc_test', accuracy_test, each_epoch)

        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
        print('the best accuracy is {}'.format(accuracy_best,'.2f'))























