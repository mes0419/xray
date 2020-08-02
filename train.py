import os
from Utils import xrayDataLoader as x_loader
import torch
from Utils.logger import log

#Logger
Tag = 'Train'

#parameters
train_batch_size = 32

#location
LOCATION_PATH = 'google_drive'

def check_dir(path):
    if path=='colab':
        train_data_dir = '/content/drive/' + 'Shared drives' + '/YS_NW/2.Data/Train/Data'
        train_coco = '/content/drive/Shared drives/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    elif path =='google_drive':
        train_data_dir = 'G:/공유 드라이브/YS_NW/2.Data/Train/Data'
        train_coco = 'G:/공유 드라이브/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    else :
        train_data_dir = '/data/jiylee/dataset/xray/Train/Data'
        train_coco = '/data/jiylee/dataset/xray/Train/Meta/CoCo/coco_rapiscan.json'

    return train_data_dir, train_coco

def train():
    train_data_dir, train_data_coco = check_dir(LOCATION_PATH)
    my_dataloader = x_loader.XrayDataLoader(root=train_data_dir, annotation=train_data_coco, batch_size=train_batch_size).get_data_loader()
    log(Tag, 'total dataset : '+str(len(my_dataloader.dataset)))

if __name__ == '__main__':
    train()
