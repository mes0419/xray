# import

import os
import torch
import torchvision
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data

# Log Tag
Tag = 'XrayDataloader'

# coalb detect logic
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from xray.Utils.logger import log
else:
    from Utils.logger import log

class XrayDataSet(data.Dataset):
    def __init__(self, root, annotation, class_name=None, transforms=None) :
        self.root = root
        self.coco = COCO(annotation)
        self.class_name = class_name
        self.transforms = transforms

        if self.class_name is not None :
            class_id = sorted(self.coco.getCatIds(catNms=self.class_name))
            self.ids = self.make_id_list(class_id)
        else :
            self.ids = self.isValid(list(sorted(self.coco.imgs.keys())))

        self.ids = list(set(self.ids))
        log(Tag, 'Dataset Created')

    def __getitem__(self, item) :
        # Own coco file
        coco = self.coco

        # Image ID
        img_id = self.ids[item]

        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)

        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)

        # Load Image
        img_info = coco.loadImgs(coco_annotation[0]["image_id"])

        # file path
        file_path = img_info[0]["path"].split('\\', maxsplit=7)[-1]
        #log(Tag, str(file_path))
        # open the input image
        image_path = os.path.join(self.root, file_path.replace('\\', '/'))
        img = Image.open(image_path)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        class_id = []
        for i in range(num_objs):
            # bbox
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            # class_id
            cid = coco_annotation[i]["category_id"]
            class_id.append(cid)

        file_name = img_info[0]["file_name"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        class_names = self.get_class_names(class_id)
        class_id = torch.as_tensor(class_id)
        image_id = coco_annotation[0]["image_id"]

        # Tensorise
        image_id = torch.tensor([image_id])

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Annotation is in dictionary format
        xray_annotation = {"boxes" : boxes, "class_id": class_id, "image_id" : image_id, 'filename' : file_name,
                           'class_names' : class_names}

        if self.transforms is not None :
            img = self.transforms(img)

        return img, xray_annotation

    def make_id_list(self, class_id) :
        # 공유폴더에 이미지가 존재하는것만 index를 만들어줌
        ids = []

        if isinstance(class_id, list):
            for id in class_id:
                find_id = self.coco.getImgIds(catIds=[id])
                ids.extend(self.isValid(find_id))
        else:
            find_id = self.coco.getImgIds(catIds=class_id)
            ids.extend(self.isValid(find_id))

        return ids

    def isValid(self, find_id) :
        ids = []
        invalid = []
        log(Tag, 'isValid check len : ' + str(len(find_id)))
        for i in find_id :
            file = self.coco.imgs[i]['path'].split('\\', maxsplit=7)[-1].replace('\\', '/')
            if os.path.isfile(os.path.join(self.root, file)):
                ids.append(i)
            else :
                invalid.append(file)
        if invalid:
            log(Tag, 'Empty file found: '+str(len(invalid)))
            if os.path.exists('./debug') is not True :
                os.makedirs('debug')
            with open('./debug/empty_file.txt', 'w') as fp :
                for s in invalid :
                    fp.write(str(s) + "\n")
        return ids

    # show all category_id and its name
    def get_object_info(self):
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)
        df_rows = []
        for c in sorted(cats, key=lambda x: x["id"]):
            #image count in our path
            path_data_cnt = len(self.make_id_list(c["id"]))
            #image count in annotation file
            anno_data_cnt = len(self.coco.getImgIds(catIds=c["id"]))
            df_rows = df_rows + [[c["id"], c["name"], path_data_cnt, anno_data_cnt]]

        return pd.DataFrame(df_rows, columns=["cat_id", "cat_nm", "path_data_cnt", "anno_data_cnt"])

    def __len__(self):
        return len(self.ids)

    def class_id_to_str(self, class_id):
        class_dict = {34:'ZippoOil', 37:'Chisel', 24:'Scissors', 30:'SupplymentaryBattery', 22:'PrtableGas',
                      36:'Plier', 15:'Knife', 17:'Lighter', 11:'Hammer', 9:'Gun', 20:'MetalPipe', 25:'Screwdriver',
                      4:'Axe', 28:'Spanner', 23:'Saw', 10:'GunParts', 1:'Aerosol', 19:'Match', 2:'Alcohol',
                      39:'Electronic cigarettes(Liquid)', 12:'HandCuffs', 41:'Throwing Knife', 32:'Thinner',
                      40:'stun gun', 38:'Electronic cigarettes', 26:'SmartPhone', 13:'HDD', 27:'SolidFuel',
                      6:'Battery', 3:'Awl', 18:'Liquid', 33:'USB', 31:'TabletPC', 29:'SSD', 21:'NailClippers',
                      16:'Laptop', 7:'Bullet', 8:'Firecracker', 5:'Bat'}

        return class_dict[class_id]

    def get_class_names(self, class_ids):
        class_name = []
        for class_id in class_ids:
            class_name.append(self.class_id_to_str(class_id))
        return class_name

def get_transform() :
    custom_transforms = [torchvision.transforms.ToTensor()]
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch) :
    return tuple(zip(*batch))

class XrayDataLoader() :
    def __init__(self, root, annotation, batch_size, class_name=None, transfroms=None) :
        self.xray_data_set = XrayDataSet(root, annotation, class_name, transfroms)
        self.batch_size = batch_size
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.xray_data_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            collate_fn=collate_fn,)

    def get_data_loader(self):
        return self.data_loader