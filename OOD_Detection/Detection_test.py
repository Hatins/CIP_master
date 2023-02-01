import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torchvision.models as md
import torchvision
import argparse
import os
import pandas as pd
from Victim_Model_Train.Model.wrn import WideResNet
device = torch.device('cuda:0')
parser = argparse.ArgumentParser(description='Testing the OOD detector',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--attack',type=str,default='IDA',choices=['Knockoff','Datafree','IDA'],help='choosing the attack forms')
parser.add_argument('--dataset',type=str,default='Cifar10',choices=['Mnist','Cifar10','Fashion_mnist','Cifar100','FOOD101'], help='choosing the dataset for victim model')
parser.add_argument('--model',type=str,default='wrn_28',choices=['resnet18','wrn_28'],help = 'choosing the model for victim model')
parser.add_argument('--num_classes',type=int,default=10,help='setting the num_classes for victim model training')
parser.add_argument('--num_workers',type=int,default=0,help='setting the num_worker for detector ')
parser.add_argument('--energy_ft',type=int,default=1,choices = [0,1],help='whether the ptfile has been handled ')
parser.add_argument('--size',type=int,default=32,help='setting the image size')
parser.add_argument('--test_batch',type=int,default=200,help ='setting the test batch')
parser.add_argument('--FPR',type=float,default = 0.8, help = 'setting the FPR value, in the testing of IDA and DQA, it was set to 0.8')
parser.add_argument('--test_data',type=str,default='DTD',choices = ['DTD','ISUN','LSUN','Noise','Place365','ID_test'],help='whether the ptfile has been handled ')
args = parser.parse_args(args=[])
if args.attack == 'IDA':
    args.type = 'half'
else:
    args.type = 'all'

if args.dataset == 'Cifar10':
    data_exp = 'Cifar100'
else:
    data_exp = '------------'
def get_ood_scores(loader,Victim_model):
    _score = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device)
            output = Victim_model(data)
            _score.append(-to_np((1 * torch.logsumexp(output / 1, dim=1))))

    return concat(_score).copy()


def get_ood_scores_single(image,Victim_model):
    image = image.unsqueeze(0)
    with torch.no_grad():
        image = image.to(device)
        output = Victim_model(image)
        score = -to_np((1*torch.logsumexp(output/1, dim=1)))
    return score

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
    Victim_model =  WideResNet(depth=28,num_classes=args.num_classes,widen_factor=10,dropRate=0.3)
elif args.model == 'wrn_40':
    Victim_model =  WideResNet(depth=40,num_classes=args.num_classes,widen_factor=2,dropRate=0.3)

num_params = sum(i.numel() for i in Victim_model.parameters() if i.requires_grad)
print('the number of parameters is {}'.format(num_params))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


if __name__ == '__main__':

    '''setting the close_set dataset'''
    test_path = r'../Victim_Model_Train/Dataset/' + '/' + args.dataset + '/' + args.dataset + '_' + 'all' + '/train'
    test_data = dset.ImageFolder(test_path, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=128, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    ood_num_examples = len(test_data) // 5
    expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

    if args.energy_ft == 0:
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


    Victim_model.load_state_dict(torch.load(pretrained_file,map_location='cuda:0'))
    Victim_model.eval()
    Victim_model = Victim_model.to(device)
    cudnn.benchmark = True

    in_score = get_ood_scores(test_loader,Victim_model)
    in_score_sort = sorted(in_score)
    FPR = in_score_sort[int((args.FPR)*len(in_score_sort))]
    # print('The FPR{} energy is '.format(int(args.FPR*100)),round(FPR,2))
    # print('The highest energy of OOD with low ES is {}'.format(round(in_score_sort[-int((0.001)*len(in_score))],2)))
    # print('The close_set average energy is',round(sum(in_score_sort) / len(in_score_sort),2)) #close_set能量平均值
    information = {'dataset':args.dataset,'FPR_energy':FPR,'open-set_energy':in_score_sort[int((0.999)*len(in_score))]}


    information_dataframe = pd.DataFrame(information,index=[0])
    write_place =  args.dataset+'_'+args.attack
    if not os.path.exists('./information.xlsx'):
        writer = pd.ExcelWriter('./information.xlsx')
        information_dataframe.to_excel(writer, write_place, float_format='%.2f')
        writer.close()
    else:
        writer = pd.ExcelWriter('./information.xlsx',mode='a',engine="openpyxl")
        wb = writer.book
        if write_place in wb.sheetnames:
            wb.remove(wb[write_place])
        information_dataframe.to_excel(writer,write_place,float_format='%.2f')
        writer.close()

    open_data_path = r'./Dataset/Open-set test/'+args.test_data+'_image_test'
    close_set_data_path = r'../Victim_Model_Train/Dataset/' + '/' + args.dataset + '/' + args.dataset + '_' + 'all' + '/train'

    ood_data = dset.ImageFolder(root=open_data_path,
                                transform=trn.Compose([trn.Resize([args.size,args.size]),
                                                       trn.RandomCrop(args.size),
                                                       transform_test]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_batch, shuffle=True,
                                             num_workers=args.num_workers, pin_memory=True)
    out_score = get_ood_scores(ood_loader,Victim_model)
    # print('The open_set avergy energy is ',round(sum(out_score)/len(out_score),2))  #open_set能量平均值


    t = 0
    s = 0
    lowest_ES = 0
    for i in range(len(out_score)):
        if out_score[i] < lowest_ES:
            lowest_ES = out_score[i]
        if out_score[i] < FPR:
            t = t +1
        if out_score[i] > in_score_sort[int((0.999)*len(in_score))]:
            s = s +1

    print('{} FPR{}:'.format(args.test_data,int(100*args.FPR)),round((t/len(out_score))*100,2),'%')
    # print('OOD high ES ratio:',(round((s/len(out_score))*100,2),'%'))
    # print('the lowest ES of OOD is {}'.format(lowest_ES))
