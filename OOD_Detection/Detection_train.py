import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import numpy as np
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from Victim_Model_Train.Model.wrn import WideResNet
import torchvision.models as md
from Utils.validation_dataset import validation_split
from  Utils.tinyimages_80mn_loader import TinyImages
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--attack',type=str,default='Datafree',choices=['Knockoff','Datafree','IDA'],help='choosing the attack forms')
parser.add_argument('--dataset',type=str,default='Mnist',choices=['Mnist','Gtsrb','Cifar10','Fashion_mnist','FOOD101'], help='choosing the dataset for victim model')
parser.add_argument('--model',type=str,default='resnet18',choices=['resnet18','wrn_28'],help = 'choosing the model for victim model')
parser.add_argument('--num_classes',type=int,default=10,help='setting the num_classes for victim model training')
parser.add_argument('--calibration', '-c', action='store_true',help='Train a model to be used for calibration. This holds out some data for validation.')
# parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--size', type=int, default=28, help='Size of image')
parser.add_argument('--ood', type=str, default='tiny_image',choices=['tinyimage','noise'], help='Size of image')
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_workers',type=int,default=0,help='setting the num_worker for detector ')

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')

#Mnist don't need  Cifar10: m_in:-25 m_out-7  Gtsrb: M_in:-25 M_out:-7
parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')
args = parser.parse_args(args=[])
state = {k: v for k, v in args._get_kwargs()}
if args.attack == 'IDA':
    args.type = 'half'
else:
    args.type = 'all'

if args.dataset == 'Cifar10':
    data_exp = 'Cifar100'
else:
    data_exp = '-------------'
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
        module.num_batches_tracked = 0
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

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
    Victim_model =  WideResNet(depth=28,num_classes=args.num_classes,widen_factor=10,dropRate=0.3)

'''setting dataset'''
train_path = r'../Victim_Model_Train/Dataset/' +'/'+ args.dataset + '/'+args.dataset+'_'+args.type+'/train'
test_path = r'../Victim_Model_Train/Dataset/' +'/'+ args.dataset + '/'+args.dataset+'_'+args.type+'/test'

train_data_in = torchvision.datasets.ImageFolder(train_path, transform=transform_train)
test_data = torchvision.datasets.ImageFolder(test_path, transform=transform_test)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

if args.ood == 'noise':
    ood_path = r'./Dataset/' + 'Open-set test/' + 'Noise'
    ood_data = torchvision.datasets.ImageFolder(ood_path, transform=transform_train)
else:
    ood_data = TinyImages(transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.ToPILImage(), torchvision.transforms.RandomCrop(args.size, padding=4),
         torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)]))

calib_indicator = ''
if args.calibration:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    calib_indicator = '_calib'

train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.oe_batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.test_bs, shuffle=False,
    num_workers=args.num_workers, pin_memory=True)

pretrained_file_path = '../Victim_Model_Train/Trained/' + args.attack
pretrained_file_list = os.listdir(pretrained_file_path)
for i in range(len(pretrained_file_list)):
    if (args.dataset in pretrained_file_list[i]) and (data_exp not in pretrained_file_list[i]):
        pretrained_file = pretrained_file_path + '/'+pretrained_file_list[i]
        print('loading pretrained file ',pretrained_file)

Victim_model.load_state_dict(torch.load(pretrained_file,map_location='cuda:0'))
Victim_model = Victim_model.to(device)

cudnn.benchmark = True  # fire on all cylinders
optimizer = torch.optim.SGD(
    Victim_model.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

def train():
    Victim_model.train()  # enter train mode
    loss_avg = 0.0

    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))

    loop_train = tqdm(zip(train_loader_in, train_loader_out), total=len(train_loader_in))
    for in_set, out_set in loop_train:
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.to(device), target.to(device)

        x = Victim_model(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(x[:len(in_set[0])], target)
        Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
        Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
        loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())

        loss.backward()
        optimizer.step()
        scheduler.step()
        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        loop_train.set_description(f'training_Epoch [{epoch + 1}/{args.epochs}]')
        loop_train.set_postfix(train_loss=loss_avg)


def tes():
    accuracy = 0
    Victim_model.eval()
    loss_avg = 0.0
    correct = 0
    loop_test = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for data, target in loop_test:
            data, target = data.to(device), target.to(device)

            # forward
            output = Victim_model(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

            loop_test.set_description(f'testing_Epoch [{epoch + 1}/{args.epochs}]')
            loop_test.set_postfix(test_loss=loss_avg / len(test_loader),test_acc =(100.*(correct / len(test_loader.dataset))))
    accuracy = 100.*(correct / len(test_loader.dataset))
    return  accuracy

if __name__ == '__main__':
    print('Beginning Training\n')
    for epoch in range(0, args.epochs):
        state['epoch'] = epoch
        begin_epoch = time.time()
        train()
        accuracy = tes()
        # Save model
        if epoch == args.epochs-1:
            torch.save(Victim_model.state_dict(),
                       os.path.join('./Trained/' + args.attack + '/energy_ft_' + args.dataset + '_' + args.model + '_epoch_' + '{}'.format(accuracy, '.2f')+ '_' + str(epoch + 1) + '.pt'))
        else:
            torch.save(Victim_model.state_dict(),
                       os.path.join('./Trained/'+args.attack+'/energy_ft_'+args.dataset+'_'+args.model+'_epoch_' + str(epoch+1) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join('./Trained/'+args.attack+'/energy_ft_'+args.dataset+'_'+args.model+'_epoch_' + str(epoch) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)
    print('{}'.format(accuracy,'.2f'))
