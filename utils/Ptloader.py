# ------------------------------------------------------------------------
# CIP: Categorical Inference Poisoning: Verifiable Defense Against Black-Box DNN Model Stealing Without Constraining Surrogate Data and Query Times
# Haitian Zhang, Guang Hua, Xinya Wang, Hao Jiang, and Wen Yang
# paper: https://ieeexplore.ieee.org/document/10042038
# -----------------------------------------------------------------------

import torch
from torch.utils.data import Dataset

def load_pt_file(path):
    with open(path, 'rb') as f:
        data = torch.load(path)
        image_list = data['image']
        probability_list = data['probability']
        if isinstance(image_list,list) == False:
            image_list = image_list.tolist()
        if isinstance(probability_list,list) == False:
            probability_list = probability_list.tolist()
        for i in range(len(probability_list)):
            probability_list[i] = torch.tensor(probability_list[i])
        image = torch.stack(image_list)
        probability = torch.stack(probability_list)

        return image, probability

class Ptloader(Dataset):
    def __init__(self, path_name,train = True,transform=None):
        super(Ptloader,self).__init__()
        self.path_name=path_name
        self.transform = transform
        if(train):
            self.image, self.probability = load_pt_file(path_name)

    def __len__(self):
        return len(self.image)

    def __getitem__(self,index):
        image = self.image[index]
        probability = self.probability[index]
        if self.transform != None:
            image = self.transform(image)
        return image, probability

