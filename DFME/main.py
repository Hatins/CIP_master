# ------------------------------------------------------------------------
# CIP: Categorical Inference Poisoning: Verifiable Defense Against Black-Box DNN Model Stealing Without Constraining Surrogate Data and Query Times
# Haitian Zhang, Guang Hua, Xinya Wang, Hao Jiang, and Wen Yang
# paper: https://ieeexplore.ieee.org/document/10042038
# -----------------------------------------------------------------------
# Modified from Data-Free Model Extraction
# Jean-Baptiste Truong, Pratyush Maini, Robert J. Walls, Nicolas Papernot
# paper: https://openaccess.thecvf.com/content/CVPR2021/html/Truong_Data-Free_Model_Extraction_CVPR_2021_paper.html
# codes: https://github.com/cake-lab/datafree-model-extraction
# ------------------------------------------------------------------------

from __future__ import print_function
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import argparse
import torch.optim as optim
import torchvision.models as md
from torch.utils.data import DataLoader
from model.gan import GeneratorA
from utils.approximate_gradients import estimate_gradient_objective, compute_gradient

to_np = lambda x: x.data.cpu().numpy()
from categorical_inference_poisoning.function import *
def get_ood_scores_batch(image,model):
    output = []
    score_list = []
    with torch.no_grad():
        for i in range(len(image)):
            image[i] = image[i].to(args.device)
            output = model(image[i])
            score = -to_np((1*torch.logsumexp(output/1, dim=1)))
            output.append(output)
            score_list.append(score[0])
        output_tensor = torch.cat([i for i in output], dim=0)
        return output_tensor,score_list

def get_ood_scores_single(image,model):
    # image = image.unsqueeze(0)
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        score = -to_np((1*torch.logsumexp(output/1, dim=1)))
    return output,score

def open_set_poisoned_batch(ood_probability,energy):
    for i in range(len(ood_probability)):
        ood_probability[i] = ood_probability[i].cpu()
        if energy[i] > args.open_set_energy:
            ood_probability[i] = torch.tensor(SPPD(ood_probability[i]))
        elif energy[i] <= args.open_set_energy and energy[i] > args.FPR:
            ood_probability[i] = compute_noise2(ood_probability[i])
        ood_probability[i] = ood_probability[i].to(device)
    return ood_probability

def student_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits =  False
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss

def measure_true_grad_norm(args, x):
    # Compute true gradient of loss wrt x
    true_grad, _ = compute_gradient(args, args.target_model, args.Surrogate_model, x, pre_x=True, device=args.device)
    true_grad = true_grad.view(-1, 3072)

    # Compute norm of gradients
    norm_grad = true_grad.norm(2, dim=1).mean().cpu()

    return norm_grad

def compute_grad_norms(generator, student):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            G_grad.append(p.grad.norm().to("cpu"))

    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            S_grad.append(p.grad.norm().to("cpu"))
    return  np.mean(G_grad), np.mean(S_grad)

def a_test(args, Surrogate_model = None, Generator = None, device = "cuda:0", test_loader = None, epoch=0):
    global file
    Surrogate_model.eval()
    Generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = Surrogate_model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    acc = correct/len(test_loader.dataset)
    return acc

def train(args, target_model, Surrogate_model,Generator,optimizer,epoch,device):
    energy_list = []
    target_model.eval()
    Surrogate_model.eval()

    optimizer_S, optimizer_G = optimizer
    for i in range(args.epoch_itrs):
        for _ in range(args.g_iter):
            noise = torch.randn((args.batch, args.noise_size)).to(device)
            optimizer_G.zero_grad() #set gradient to 0 in each epoch
            Generator.train()

            fake = Generator(noise, pre_x=args.approx_grad)
            approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, target_model, Surrogate_model, fake,
                                                                    epsilon=args.grad_epsilon, m=args.grad_m,
                                                                    num_classes=args.num_classes,
                                                                    device=device, pre_x=True)
            fake.backward(approx_grad_wrt_x)
            optimizer_G.step()

            if i == 0 and args.rec_grad_norm:
                x_true_grad = measure_true_grad_norm(args, fake)

        for _ in range(args.d_iter):
            noise = torch.randn((args.batch, args.noise_size)).to(device)
            fake = Generator(noise).detach()
            optimizer_S.zero_grad()
            # with torch.no_grad():
            #     t_logit = target_model(fake)

            if args.loss == "l1" and args.no_logits:
                t_logit,energy= get_ood_scores_single(fake, target_model)
                energy_list.extend(energy)
                t_soft = F.softmax(t_logit,dim=1).detach()
                if args.poison == True and args.method == 'CIP':
                    t_soft = open_set_poisoned_batch(t_soft,energy)
                elif args.poison == True and args.method == 'DP':
                    t_soft = compute_noise2(t_soft)
                t_logit = torch.log(t_soft)
                # t_logit = F.log_softmax(t_logit, dim=1).detach()

                if args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            s_logit = Surrogate_model(fake)
            loss_S = student_loss(args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()

        # Log Results
        if i % args.log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f}')

            if args.rec_grad_norm and i == 0:

                G_grad_norm, S_grad_norm = compute_grad_norms(Generator, Surrogate_model)


        # update query budget
        args.query_times -= args.cost_per_iteration

        if args.query_times < args.cost_per_iteration:
            return energy_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DFME')
    parser.add_argument('--query_times',type=int,default=6*(10**6),help = 'setting the query_times')
    parser.add_argument('--attack', type=str, default='DFME',help='setting the attack form')
    parser.add_argument('--Surrogate_model',type=str,default='resnet18',choices=['wrn28','resnet18'],help='setting the surrogate model')
    parser.add_argument('--seed',type=int,default=1,metavar='S',help='setting the random seed')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--num_classes',type=int,default=10,help='setting the number of classes')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FMNIST', 'CIFAR10'], help='choosing the dataset')
    parser.add_argument('--target_model',type=str,default='resnet18',choices=['wrn28','resnet18'],help='setting the target model')
    parser.add_argument('--pt_file',type=str,default='',help='setting the target model')
    parser.add_argument('--batch',type=int,default=128,help='setting the batch_size')
    parser.add_argument('--num_workers',type=int,default=0,help='setting the num_workers')
    parser.add_argument('--size',type=int,default=28,help='setting the image size')
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float,
                        help="Percentage epochs at which to take next step")
    parser.add_argument('--poison',type=bool,default=True,help='whether poisoning')
    parser.add_argument('--method', type=str, default='DP',choices=['DP','CIP'])
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'] )
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])
    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--grad_epsilon', type=float, default=1e-3)
    '''GAN'''
    parser.add_argument('--log_dir', type=str, default="results")
    parser.add_argument('--noise_size',type=int,default=256,help='size of random noise input to generator')
    parser.add_argument('--approx_grad', type=int, default=1, help='Always set to 1')
    parser.add_argument('--g_iter', type=int, default=1, help="Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help="Number of discriminator iterations per epoch_iter")
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
    parser.add_argument('--rec_grad_norm', type=int, default=1)
    args = parser.parse_args()

    file_path = '../OOD_Detection/information.xlsx'
    sheet_name = args.dataset + '_' + args.attack
    file = pd.read_excel(file_path, sheet_name=sheet_name)
    FPR = float(file.loc[0]['FPR_energy'])
    open_set_energy = float(file.loc[0]['open-set_energy'])
    args.FPR = FPR
    args.open_set_energy = open_set_energy

    args.G_activation = torch.tanh
    '''setting the random seed'''
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    '''setting the transform'''
    if args.dataset == 'MNIST':
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
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    elif args.dataset == 'FMNIST':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])

    device = torch.device("cuda:0")
    args.device = device
    if args.target_model == 'wrn28':
        target_model = WideResNet(depth=28,num_classes=args.num_classes,widen_factor=10,dropRate=0.3)
    elif args.target_model == 'resnet18':
        target_model = md.resnet18(num_classes = args.num_classes)

    target_model.load_state_dict(torch.load(args.pt_file,map_location=device))
    target_model.eval()
    target_model = target_model.to(device)

    print('the target is from '+ args.pt_file)
    print('the target model is ' + args.target_model)
    print('the Surrogate model is ' + args.Surrogate_model)

    '''target model testing'''
    correct_number = 0

    test_path = r'../model_train/Dataset/' + args.dataset + '/' + args.dataset + '_all' + '/test'
    test_set = torchvision.datasets.ImageFolder(test_path, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = target_model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct_number += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct_number / len(test_loader.dataset)
    print('target - Test set: Accuracy: {}/{} ({:.4f}%)'.format(correct_number, len(test_loader.dataset),accuracy))

    '''setting the surrogate model and gan'''
    if args.Surrogate_model == 'resnet18':
        Surrogate_model = md.resnet18(num_classes=args.num_classes)
    elif args.Surrogate_model == 'wrn28':
        Surrogate_model = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10, dropRate=0.3)

    Generator = GeneratorA(nz=args.noise_size,nc=3,img_size=args.size,activation=torch.tanh)

    Surrogate_model = Surrogate_model.to(device)
    Generator = Generator.to(device)

    args.Generator = Generator
    args.Surrogate_model = Surrogate_model
    args.target_model = target_model

    args.cost_per_iteration = args.batch * (args.g_iter * (args.grad_m + 1) + args.d_iter)

    number_epochs = args.query_times // (args.cost_per_iteration * args.epoch_itrs) + 1

    print('Total query times is {}'.format(args.query_times))
    print('cost per iterations: ', args.cost_per_iteration)
    print('Total number of epoch: ', number_epochs)

    optimizer_S = optim.SGD(Surrogate_model.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(Generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    best_acc = 0
    acc_list = []

    for epoch in range(1, number_epochs + 1):
        # Train
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

        energy_list = train(args, target_model=target_model, Surrogate_model=Surrogate_model,Generator=Generator,
              optimizer=[optimizer_S,optimizer_G],epoch=epoch,device=args.device)

        acc = a_test(args, Surrogate_model=Surrogate_model, Generator=Generator, device=args.device, test_loader=test_loader, epoch=epoch)

        if acc>best_acc:
            best_acc = acc

        print('the best_acc is {}'.format(best_acc))
    information = {'energy':energy_list}
    information_dataframe = pd.DataFrame(information)

















