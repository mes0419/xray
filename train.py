######[IMPORT MODULE]#######
from Utils import xrayDataLoader as x_loader
from Utils.logger import log

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

######[GOLBAL VLAUE]#######
# Logger
Tag = 'Train'

# parameters
train_batch_size = 32

def check_dir(path):
    if path == 'colab':
        train_data_dir = '/content/drive/' + 'Shared drives' + '/YS_NW/2.Data/Train/Data'
        train_coco = '/content/drive/Shared drives/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    elif path == 'google_drive':
        train_data_dir = 'G:/공유 드라이브/YS_NW/2.Data/Train/Data'
        train_coco = 'G:/공유 드라이브/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    else:
        train_data_dir = '/data/jiylee/dataset/xray/Train/Data'
        train_coco = '/data/jiylee/dataset/xray/Train/Meta/CoCo/coco_rapiscan.json'

    return train_data_dir, train_coco


def get_data_loader():
    train_data_dir, train_data_coco = check_dir(LOCATION_PATH)
    _data_loader = x_loader.XrayDataLoader(root=train_data_dir, annotation=train_data_coco,
                                          batch_size=train_batch_size).get_data_loader()
    log(Tag, 'get_data_loader: loader make complete, total dataset : ' + str(len(_data_loader.dataset)))
    return _data_loader

# show image

def check_dataset(data_set, num_of_item):
    sample = random.sample(range(0, len(data_set) - 1), num_of_item)
    for i in sample:
        plt.figure()
        sample_img = data_set[i][0]
        sample_anno = data_set[i][1]
        bb = np.array(sample_anno["boxes"], dtype=np.float32)
        filename = np.array(sample_anno["filename"])
        for j in range(len(bb)):
            ImageDraw.Draw(sample_img).rectangle(bb[j], outline='red', width = 3)
        np_sample = np.array(sample_img)
        plt.title(filename)
        plt.imshow(np_sample)
        plt.show()
    return sample

if __name__ == '__main__':
    # location
    LOCATION_PATH = 'google_drive'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader = get_data_loader()
    sample = check_dataset(data_loader.dataset, 10)

