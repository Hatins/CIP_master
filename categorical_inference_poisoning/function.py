# ------------------------------------------------------------------------
# CIP: Categorical Inference Poisoning: Verifiable Defense Against Black-Box DNN Model Stealing Without Constraining Surrogate Data and Query Times
# Haitian Zhang, Guang Hua, Xinya Wang, Hao Jiang, and Wen Yang
# paper: https://ieeexplore.ieee.org/document/10042038
# ------------------------------------------------------------------------
# DAWN: DAWN: Dynamic Adversarial Watermarking of Neural Networks
# Sebastian Szyller, Buse Gul Atli, Samuel Marchal, N. Asokan
# paper: https://dl.acm.org/doi/abs/10.1145/3474085.3475591
# -----------------------------------------------------------------------
# DP: Defending against neural network model stealing attacks using deceptive perturbations
# Taesung Lee, Benjamin Edwards, Ian Molloy, Dong Su
# paper: https://ieeexplore.ieee.org/document/8844598
# -----------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..")))
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from Model.wrn import WideResNet
from PIL import Image
from tqdm import tqdm
import pandas as pd
import heapq
import random

device = torch.device('cuda:0')
to_np = lambda x: x.data.cpu().numpy()
random.seed(0)


def index_sort(p):
    p = np.array(p)
    p_index = []
    for i in range(len(p)):
        data = p[i]
        p_except = np.delete(p,i)
        number = 0
        for j in range(len(p_except)):
            if data > p_except[j]:
                number = number + 1
            elif data == p_except[j]:
                if number in p_index:
                    number = number + 1
                else:
                    number = number
        p_index.append(number)
    return p_index

def upside_down(p):
    p = np.array(p)
    p_sort_index = index_sort(p)
    p_sort = sorted(p,reverse=True)
    p_out = []
    for i in range(len(p)):
        p_out.append(p_sort[p_sort_index[i]])
    p_out = np.array(p_out)
    return p_out

def wgn(x, snr):
   snr = 10**(snr/10.0)
   xpower = np.sum(x**2)/len(x)
   npower = xpower / snr
   return np.absolute(np.random.randn(len(x)) * np.sqrt(npower))

def SPPD(p):
    p = np.array(p)
    p_max = p.argmax()
    random.shuffle(p)
    while p.argmax() == p_max:
        random.shuffle(p)
    p = (wgn(p,10)+p)
    p = p/sum(p)
    return p

def convert(p):
    p = np.array(p)
    p = p + 10e-30
    p = 1 / p
    p = p / sum(p)
    return p

def smooth(p):
    p = np.array(p)
    p = pow(p, 1 / 30)
    p = p / np.sum(p)
    return p

def no_poison(p):
    return p

def reliability(p_list):
    N = len(p_list)
    max_index = np.argmax(p_list)
    p_max = p_list[max_index]
    p_exceptpmax = np.delete(p_list, max_index)
    std = np.std(p_exceptpmax)
    R = (pow(p_max, 2) - (1/N)) / (std + (1-pow(1/N,2)))
    return R

def get_ood_scores_single(image, model):
    image = image.unsqueeze(0)
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        score = -to_np((1 * torch.logsumexp(output / 1, dim=1)))
    return output, score[0]

def close_set_probability_poisoned(id_probability):
    id_probability = np.array(id_probability)
    label_raw = id_probability.argmax()
    id_probability = id_probability + 10e-30
    temp_probability = (id_probability.min() / 100)
    for single_probability in range(len(id_probability)):
        if single_probability == id_probability.argmin():
            id_probability[single_probability] = temp_probability
        else:
            id_probability[single_probability] = id_probability[single_probability] + temp_probability * 11
    id_probability = 1 / id_probability
    id_probability = id_probability / np.sum(id_probability)
    random.shuffle(id_probability)
    while label_raw == id_probability.argmax():
        random.shuffle(id_probability)
    return id_probability

def sort_by_key(d):
    return sorted(d.items(), key=lambda k: k[0])

def sigmoid(z):
    return torch.sigmoid(z)

def inv_sigmoid(p):
    assert (p >= 0.).any()
    assert (p <= 1.).any()
    return torch.log(p / (1 - p))

def reverse_sigmoid(y, beta, gamma):
    return beta * (sigmoid(gamma * inv_sigmoid(y)) - 0.5)

def compute_noise2(y_v):
    beta = 0.9
    gamma = 0.6
    y_prime = y_v - reverse_sigmoid(y_v, beta, gamma)
    y_prime /= y_prime.sum()

    return y_prime

def compute_noise(args,y_v):
    beta = args.B
    gamma = args.Y
    y_prime = y_v - reverse_sigmoid(y_v, beta, gamma)
    y_prime /= y_prime.sum()

    return y_prime

def DAWN_shuffle(target):
    for change in range(len(target) - 1, 0, -1):
        lower = random.randint(0, change-1)
        target[lower], target[change] = target[change], target[lower]

def DAWN(p,numclass):
    p = list(p)
    number = int(pow(numclass,(1/2)))
    max_num_index_list = list(map(p.index, heapq.nlargest(number,p)))
    p = np.array(p)
    row_index = max_num_index_list.copy()
    DAWN_shuffle(max_num_index_list)
    new_index = max_num_index_list
    p_copy = p.copy()
    for i in range(len(max_num_index_list)):
        p[row_index[i]] = p_copy[new_index[i]]
    return p

def poison_fun_tensor(args,model, image_tensor, transform_test,
                      attack, ratio, dataset, trigger_path, poison_way, poison=True):
    # image_tensor is the input image list or tensor
    # trigger number = close_set_num * ratio
    # poison decide whether the input was poisoned

    if poison == False:
        ratio = 0
    # using for compute the ground true labels
    if dataset == 'Cifar10':
        interval = 2500
    elif dataset == 'Cifar100':
        interval = 200
    else:
        interval = 500

    model.eval()
    model = model.to(device)
    index = 0
    loop_save = tqdm(enumerate(image_tensor),
                     total=len(image_tensor))
    loop_save.set_description(f'saving...')

    '''ood_high_score_processing'''
    ood_high_score_image_list = []
    ood_high_score_name_list = []
    ood_high_score_list = []
    ood_high_socre_probability_list = []
    ood_high_socre_probability_poisoned_list = []
    ood_high_socre_index_list = []
    '''ood_high_score_processing'''

    '''ood_classification_processing'''
    ood_classification_image_list = []
    ood_classification_name_list = []
    ood_classification_score_list = []
    ood_classification_probability_list = []
    ood_classification_R_list = []
    ood_classification_index_list = []
    '''ood_classification_processing'''

    '''id_classification_processing'''
    id_classification_image_list = []
    id_classification_name_list = []
    id_classification_score_list = []
    id_classification_probability_list = []
    id_classification_R_list = []
    id_classification_index_list = []
    label_prediction_list = []
    label_true_list = []
    '''id_classification_processing'''

    for index, data in loop_save:
        image_save = image_tensor[index]
        if transform_test == None:
            image = image_tensor[index]
        else:
            image = transform_test(image_tensor[index])
        image = image.to(device)
        output, ood_score = get_ood_scores_single(image, model)
        _, predicted = torch.max(output.data, 1)
        p_list = F.softmax(output.data, dim=1).to('cpu')
        p_list_np = p_list.numpy()
        p_list_np = p_list_np[0]
        R = reliability(p_list_np)
        image_save = image_save.to('cpu')

        file_path = '../OOD_Detection/information.xlsx'
        sheet_name = dataset + '_' + attack
        file = pd.read_excel(file_path, sheet_name=sheet_name)
        FPR = float(file.loc[0]['FPR_energy'])
        Open_set_energy = float(file.loc[0]['open-set_energy'])
        if ood_score > Open_set_energy:  # OOD-high-suspectable samples
            ood_high_score_image_list.append(image_save.squeeze())
            ood_high_score_name_list.append(data)
            ood_high_score_list.append(ood_score)
            ood_high_socre_probability_list.append(p_list_np)
            if poison == False:
                ood_high_socre_probability_poisoned_list.append(p_list_np)
            else:
                ood_high_socre_probability_poisoned_list.append(poison_way(p_list_np))
            index = index + 1
            ood_high_socre_index_list.append(index)
        elif ood_score > FPR:  # FPR95  OOD-low-susceptible samples
            ood_classification_image_list.append(image_save.squeeze())
            ood_classification_name_list.append(data)
            ood_classification_score_list.append(ood_score)
            if poison == False:
                ood_classification_probability_list.append(p_list_np)
            else:
                ood_classification_probability_list.append(np.array(compute_noise2(torch.tensor(p_list_np))))
            ood_classification_R_list.append(R)
            index = index + 1
            ood_classification_index_list.append(index)
        else:
            label_prediction_list.append(p_list_np.argmax())
            id_classification_image_list.append(image_save.squeeze())
            id_classification_name_list.append(data)
            id_classification_score_list.append(ood_score)
            id_classification_probability_list.append(p_list_np)
            id_classification_R_list.append(R)
            label_true_list.append(int((index - 1) / interval))
            index = index + 1
            id_classification_index_list.append(index)


    print('==> the number of OOD-high-suspectible samples is {}'.format(len(ood_high_socre_index_list)))
    print('==> the number of OOD-low-suspectible samples is {}'.format(len(ood_classification_index_list)))
    print('==> the number of close-set samples is {}'.format(len(id_classification_index_list)))
    ood_high_score_information = {'index': ood_high_socre_index_list, 'name': ood_high_score_name_list,
                                  'image': ood_high_score_image_list, 'score': ood_high_score_list,
                                  'probability': ood_high_socre_probability_list,
                                  'poison_probability': ood_high_socre_probability_poisoned_list}
    ood_high_score_information_dataframe = pd.DataFrame(ood_high_score_information)

    ood_classification_information = {'index': ood_classification_index_list, 'name': ood_classification_name_list,
                                      'image': ood_classification_image_list, 'score': ood_classification_score_list,
                                      'probability': ood_classification_probability_list,
                                      'reliability': ood_classification_R_list}
    ood_classification_information_dataframe = pd.DataFrame(ood_classification_information)

    id_classification_information = {'index': id_classification_index_list, 'name': id_classification_name_list,
                                     'image': id_classification_image_list, 'score': id_classification_score_list,
                                     'probability': id_classification_probability_list,
                                     'reliability': id_classification_R_list,
                                     'prediction_label': label_prediction_list,
                                     'ground_true':label_true_list}
    id_classification_information_dataframe = pd.DataFrame(id_classification_information)

    id_classification_information_dataframe = id_classification_information_dataframe.sort_values(by="reliability",
                                                                                                  ascending=True)
    id_classification_information_dataframe = id_classification_information_dataframe.reset_index(drop=True)
    id_classification_probability_poisoned_list = id_classification_information_dataframe['probability'].copy()
    if ratio <= 1 :
        number = round(ratio * len(id_classification_probability_poisoned_list))
    else:
        number = int(ratio)
    print('setting the number of trriger is {}'.format(number))
    for item in range(number):
        # id_classification_probability_poisoned_list[item]= np.float64(id_classification_probability_poisoned_list[item])
        id_classification_probability_poisoned_list[item] = close_set_probability_poisoned(
            id_classification_probability_poisoned_list[item])

    id_classification_information_dataframe['poison_probability'] = id_classification_probability_poisoned_list

    writer = pd.ExcelWriter(dataset + '.xlsx')
    ood_high_score_information_dataframe.to_excel(writer, 'ood_high_score', float_format='%.2f')
    ood_classification_information_dataframe.to_excel(writer, 'ood_classification', float_format='%.2f')
    id_classification_information_dataframe.to_excel(writer, 'id_classification', float_format='%.2f')

    image_id_classification_list = id_classification_information_dataframe['image'].copy()
    trigger_image_list = image_id_classification_list[:number]
    probability_ood_high_score_list = ood_high_score_information_dataframe['poison_probability'].copy()
    probability_ood_classification_list = ood_classification_information_dataframe['probability'].copy()
    probability_id_classification_list = id_classification_information_dataframe['poison_probability'].copy()
    trigger_probability_list = probability_id_classification_list[:number]
    index_ood_high_score_list = ood_high_score_information_dataframe['index'].copy()
    index_ood_classification_list = ood_classification_information_dataframe['index'].copy()
    index_id_classification_list = id_classification_information_dataframe['index'].copy()

    probability_list = pd.concat(
        [probability_ood_high_score_list, probability_ood_classification_list, probability_id_classification_list])
    index_list = pd.concat([index_ood_high_score_list, index_ood_classification_list, index_id_classification_list])
    dictionary = dict(zip(index_list, probability_list))
    writer.save()
    result = dict(sort_by_key(dictionary))
    result = list(result.values())
    result = torch.cat([torch.tensor(i).unsqueeze(0) for i in result], dim=0)

    if (trigger_path != None):
        torch.save({'image': trigger_image_list, 'probability': trigger_probability_list},
                   trigger_path)

    return result
def DAWN_poison(model, image_tensor, transform_test,
                      attack, ratio, dataset, trigger_path, poison=True):

    if dataset == 'Cifar10':
        num_classes = 10
        interval = 2500
    elif dataset == 'Cifar100':
        num_classes = 100
        interval = 200
    else:
        num_classes = 101
        interval = 500

    model.eval()
    model = model.to(device)
    loop_save = tqdm(enumerate(image_tensor),
                     total=len(image_tensor))
    loop_save.set_description(f'saving...')

    '''id_classification_processing'''
    id_classification_image_list = []
    id_classification_name_list = []
    id_classification_score_list = []
    id_classification_probability_list = []
    id_classification_R_list = []
    id_classification_index_list = []
    label_prediction_list = []
    label_true_list = []
    '''id_classification_processing'''

    for index, data in loop_save:
        image_save = image_tensor[index]
        if transform_test == None:
            image = image_tensor[index]
        else:
            image = transform_test(image_tensor[index])
        image = image.to(device)
        output, ood_score = get_ood_scores_single(image, model)
        _, predicted = torch.max(output.data, 1)
        p_list = F.softmax(output.data, dim=1).to('cpu')
        p_list_np = p_list.numpy()
        p_list_np = p_list_np[0]
        R = reliability(p_list_np)
        image_save = image_save.to('cpu')

        file_path = '../OOD_Detection/information.xlsx'
        sheet_name = dataset + '_' + attack
        file = pd.read_excel(file_path, sheet_name=sheet_name)
        FPR = float(file.loc[0]['FPR_energy'])
        Open_set_energy = float(file.loc[0]['open-set_energy'])

        label_prediction_list.append(p_list_np.argmax())
        id_classification_image_list.append(image_save.squeeze())
        id_classification_name_list.append(data)
        id_classification_score_list.append(ood_score)
        id_classification_probability_list.append(p_list_np)
        id_classification_R_list.append(R)
        label_true_list.append(int((index - 1) / interval))
        index = index + 1
        id_classification_index_list.append(index)


    id_classification_information = {'index': id_classification_index_list, 'name': id_classification_name_list,
                                     'image': id_classification_image_list, 'score': id_classification_score_list,
                                     'probability': id_classification_probability_list,
                                     'reliability': id_classification_R_list,
                                     'prediction_label': label_prediction_list,
                                     'ground_true':label_true_list}

    id_classification_information_dataframe = pd.DataFrame(id_classification_information)
    id_classification_probability_poisoned_list = id_classification_information_dataframe['probability'].copy()
    id_classification_trigger_name_list = id_classification_information_dataframe['name'].copy()
    if ratio <= 1 :
        number = round(ratio * len(id_classification_probability_poisoned_list))
    else:
        number = int(ratio)
    print('setting the number of trriger is {}'.format(number))
    poison_index_list = random.sample(range(0,index-1),number)
    if poison == True:
        for item in poison_index_list:
            id_classification_probability_poisoned_list[item] = DAWN(
                id_classification_probability_poisoned_list[item],num_classes)
            id_classification_trigger_name_list[item] = 'trigger_{}'.format(item)

    id_classification_information_dataframe['poison_probability'] = id_classification_probability_poisoned_list
    id_classification_information_dataframe['name'] = id_classification_trigger_name_list
    writer = pd.ExcelWriter('DAWN_' + dataset + '.xlsx')
    id_classification_information_dataframe.to_excel(writer, 'id_classification', float_format='%.2f')

    image_id_classification_list = list(id_classification_information_dataframe['image'].copy())
    trigger_image_list = []
    for i in poison_index_list:
        trigger_image_list.append(image_id_classification_list[i])

    probability_id_classification_list = list(id_classification_information_dataframe['poison_probability'].copy())
    trigger_probability_list = []
    for i in poison_index_list:
        trigger_probability_list.append(probability_id_classification_list[i])

    index_id_classification_list = id_classification_information_dataframe['index'].copy()
    dictionary = dict(zip(index_id_classification_list, probability_id_classification_list))
    writer.save()
    result = dict(sort_by_key(dictionary))
    result = list(result.values())
    result = torch.cat([torch.tensor(i).unsqueeze(0) for i in result], dim=0)

    if (trigger_path != None):
        torch.save({'image': trigger_image_list, 'probability': trigger_probability_list},
                   trigger_path)

    return result
def DP_poison(model, image_tensor, transform_test,poison=True):

    model.eval()
    model = model.to(device)
    loop_save = tqdm(enumerate(image_tensor),
                     total=len(image_tensor))
    loop_save.set_description(f'saving...')

    '''id_classification_processing'''
    classification_probability_list = []
    '''id_classification_processing'''

    for index, data in loop_save:
        image_save = image_tensor[index]
        if transform_test == None:
            image = image_tensor[index]
        else:
            image = transform_test(image_tensor[index])

        image = image.to(device)
        output,_ = get_ood_scores_single(image, model)
        _, predicted = torch.max(output.data, 1)
        p_list = F.softmax(output.data, dim=1).to('cpu')
        p_list_np = p_list.numpy()
        p_list_np = p_list_np[0]
        image_save = image_save.to('cpu')
        if poison == True:
            classification_probability_list.append(compute_noise2(torch.tensor(p_list_np)))
        else:
            classification_probability_list.append(torch.tensor(p_list_np))
    classification_probability_list = torch.stack([i for i in classification_probability_list], dim=0)
    return classification_probability_list