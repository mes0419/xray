"""colab check"""
try :
    import google.colab
    IN_COLAB = True
except :
    IN_COLAB = False

if IN_COLAB :
    from xray.Utils.logger import log

else :
    from Utils.logger import log

import json
from collections import defaultdict
from tqdm import tqdm
import os
from pycocotools.coco import COCO

Tag = 'coco_yolo_txt'

def check_dir(path) :
    if path == 'colab':
        images_dir_path = '/content/drive/' + 'Shared drives' + '/YS_NW/2.Data/Train/Data'
        json_file_path = '/content/drive/Shared drives/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    elif path == 'google_drive':
        images_dir_path = 'G:/공유 드라이브/YS_NW/2.Data/Train/Data'
        json_file_path = 'G:/공유 드라이브/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    else :
        images_dir_path = 'G:/공유 드라이브/YS_NW/2.Data/Train/Data'
        json_file_path = 'C:\pytorch-YOLOv4-master\pytorch-YOLOv4-master/coco_rapiscan.json'

    return images_dir_path, json_file_path

"""hyper parameters"""

images_dir_path, json_file_path = check_dir('google_drive')

output_path = 'train.txt'

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
coco = COCO(json_file_path)
with open(json_file_path, encoding='utf-8') as f :
    data = json.load(f)

"""generate labels"""
images = data['images']
annotations = data['annotations']

for ant in tqdm(annotations) :
    id = ant['image_id']
    # name = os.path.join(images_dir_path, images[id]['file_name'])
    ann_ids = coco.getAnnIds(imgIds=id)
    coco_annotation = coco.loadAnns(ann_ids)
    img_info = coco.loadImgs(coco_annotation[0]["image_id"])
    cat = coco.getCatIds(catIds=id)
    file_path = img_info[0]["path"].split('\\', maxsplit=7)[-1]
    if os.path.isfile(os.path.join(images_dir_path, file_path)) is False:
        log(Tag , 'empty file : '+str(file_path))
        continue
    name = os.path.join(images_dir_path, file_path).replace('\\', '/')
    name_box_id[name].append([ant['bbox'], ant['category_id']])

"""write to txt"""
with open(output_path, 'w', encoding='utf-8') as f:
    for key in tqdm(name_box_id.keys()) :
        f.write(key)
        box_infos = name_box_id[key]
        for info in box_infos :
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            box_info = " %d,%d,%d,%d,%d" % (
                x_min, y_min, x_max, y_max, int(info[1]))
            f.write(box_info)
        f.write('\n')
f.close()
