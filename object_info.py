from Utils import xrayDataLoader as x_loader
from Utils.logger import log

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

def show_class_info():
    train_data_dir, train_data_coco = check_dir(LOCATION_PATH)

    dataset = x_loader.XrayDataSet(root=train_data_dir, annotation=train_data_coco)
    class_list =dataset.get_object_info()

    return class_list

if __name__ == '__main__':
    # location
    LOCATION_PATH = 'google_drive'
    class_info = show_class_info()

class_info
