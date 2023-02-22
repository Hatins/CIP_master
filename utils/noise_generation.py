# ------------------------------------------------------------------------
# CIP: Categorical Inference Poisoning: Verifiable Defense Against Black-Box DNN Model Stealing Without Constraining Surrogate Data and Query Times
# Haitian Zhang, Guang Hua, Xinya Wang, Hao Jiang, and Wen Yang
# paper: https://ieeexplore.ieee.org/document/10042038
# -----------------------------------------------------------------------
import os, sys
from PIL import Image
import numpy as np

def random_noise(width, height, nc):

    img = (np.random.rand(width, height, nc) * 255).astype(np.uint8)
    if nc == 3:
        img = Image.fromarray(img, mode='RGB')
    elif nc == 1:
        img = Image.fromarray(np.squeeze(img), mode='L')
    else:
        raise ValueError(f'Input nc should be 1/3. Got {nc}.')
    return img

def generate_noise(path, N):
    for i in range(N):
        image = random_noise(32, 32, 3)
        image.save(path + '/' 'N{}.png'.format(i + 1))
        print('The {}th picture has saved'.format(i))

path = './noise'
N = 50
generate_noise(path, N)
