######[IMPORT MODULE]#######
from Utils import xrayDataLoader as x_loader
from Utils.logger import log

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

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
def show_image(sample_img, sample_anno):
    plt.imshow(sample_img)

    # bbox
    bb = np.array(sample_anno["boxes"], dtype=np.float32)
    for j in range(len(bb)):
        line = plt.Rectangle((bb[j][0], bb[j][1]), bb[j][2] - bb[j][0], bb[j][3] - bb[j][1], color="red", fill=False,
                             lw=1)
        plt.gca().add_patch(line)

    return plt

def check_dataset(data_loader):
    sample = random.sample(range(0, len(data_loader) - 1), 3)
    for i in sample:

        sample_img = np.array(data_loader[i][0].permute(1, 2, 0), dtype=np.float32)
        sample_anno = data_loader[i][1]

        show_image(sample_img, sample_anno)
        plt.show()

        if i == 3:
            break

if __name__ == '__main__':
    # location
    LOCATION_PATH = 'google_drive'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader = get_data_loader()
    check_dataset(data_loader)

    #test111
