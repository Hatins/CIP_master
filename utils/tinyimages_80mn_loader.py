# ------------------------------------------------------------------------
# CIP: Categorical Inference Poisoning: Verifiable Defense Against Black-Box DNN Model Stealing Without Constraining Surrogate Data and Query Times
# Haitian Zhang, Guang Hua, Xinya Wang, Hao Jiang, and Wen Yang
# paper: https://ieeexplore.ieee.org/document/10042038
# -----------------------------------------------------------------------
# Modified from Energy-based Out-of-distribution Detection
# Weitang Liu, Xiaoyun Wang, John D. Owens, Yixuan Li
# paper: https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html
# codes: https://github.com/wetliu/energy_ood
# ------------------------------------------------------------------------


import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import numpy as np
import torch
from bisect import bisect_left


class TinyImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=True):

        data_file = open(r'Dataset/tiny_images.bin', "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open(r'./80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

            # bisection search option
            # self.cifar_idxs = tuple(sorted(self.cifar_idxs))
            #
            # def binary_search(x, hi=len(self.cifar_idxs)):
            #     pos = bisect_left(self.cifar_idxs, x, 0, hi)  # find insertion position
            #     return True if pos != hi and self.cifar_idxs[pos] == x else False
            #
            # self.in_cifar = binary_search

    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 79302017
