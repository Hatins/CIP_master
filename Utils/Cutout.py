import torch
import numpy as np


class Cutout(object):
    # 随机在图像中取一个mask的部分
    # n_holes表示在一张图像中裁剪几个patch
    # length表示一个patch的长度有几个像素
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            # 保证mask不超过图片的范围
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img