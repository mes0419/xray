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
    def __init__(self, root, annotation, class_name=None, img_type=None, flag=None, transforms=None) :
        self.root = root
        self.coco = COCO(annotation)
        self.class_name = class_name
        self.img_type = img_type
        self.flag = flag
        self.transforms = transforms

        if self.class_name is not None :
            class_id = sorted(self.coco.getCatIds(catNms=self.class_name))
            self.ids = self.make_id_list(class_id, self.class_name, self.img_type, self.flag)
        else :
            self.ids = self.isValid(list(sorted(self.coco.imgs.keys())), class_name=None, img_type=self.img_type, flag=self.flag)

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

        # file_name = img_info[0]["file_name"]
        file_name = file_path
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

    def make_id_list(self, class_id, class_name, img_type, flag) :
        # 공유폴더에 이미지가 존재하는것만 index를 만들어줌
        ids = []

        if isinstance(class_id, list):
            for id in class_id:
                find_id = self.coco.getImgIds(catIds=[id])
                ids.extend(self.isValid(find_id, class_name, img_type, flag))
        else:
            find_id = self.coco.getImgIds(catIds=class_id)
            ids.extend(self.isValid(find_id, class_name, img_type, flag))

        return ids

    def isValid(self, find_id, class_name, img_type, flag):
        ids = []
        invalid = []
        log(Tag, 'isValid check len : ' + str(len(find_id)))
        for i in find_id :
            if flag == "1":
                class_name_anno = self.coco.imgs[i]['path'].split('\\')[7]
                img_type_anno = self.coco.imgs[i]['path'].split('\\')[8]

                if class_name_anno in class_name and img_type_anno in self.get_img_type(img_type):
                    file = self.coco.imgs[i]['path'].split('\\', maxsplit=7)[-1].replace('\\', '/')
                    if os.path.isfile(os.path.join(self.root, file)):
                        ids.append(i)
                    else :
                        invalid.append(file)
            else:
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

    def get_img_type(self, img_type):

        img_type_dict = {"SD":"Single_Default"
                       ,"SO":"Single_Other"
                       ,"MC":"Multiple_Categories"
                       ,"MO":"Multiple_Other"
        }
        return [img_type_dict[x] for x in img_type]

    # show all category_id and its name
    def get_object_info(self):
        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)
        df_rows = []
        for cat in sorted(cats, key=lambda x: x["id"]):
            #image count in our path
            path_data = self.make_id_list(cat["id"], class_name=None, img_type=None, flag=None)
            path_data_cnt = len(path_data)
            #image count in annotation file
            anno_data = self.coco.getImgIds(catIds=cat["id"])
            anno_data_cnt = len(anno_data)
            sd, so, mc, mo = self.get_type_count(path_data)
            df_rows = df_rows + [[cat['id'], cat['name'], anno_data_cnt, path_data_cnt, sd, so, mc, mo]]

        return pd.DataFrame(df_rows, columns=['ID', "Name", 'ANNO_TOTAL', 'PATH_TOTAL', 'Single_Default', 'Single_Other',
                                              'Multiple_Categories', 'Multiple_Other'])

    def get_type_count(self, anno_data):
        single_Default = 0
        single_Other = 0
        multiple_Categories = 0
        multiple_Other = 0
        for anno in anno_data :
            imgs = self.coco.loadImgs(anno)
            if 'Single_Default' in imgs[0]['path'] :
                single_Default += 1
            elif 'Single_Other' in imgs[0]['path'] :
                single_Other += 1
            elif 'Multiple_Categories' in imgs[0]['path'] :
                multiple_Categories += 1
            else:
                multiple_Other += 1
        return single_Default, single_Other, multiple_Categories, multiple_Other

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
    def __init__(self, root, annotation, batch_size, class_name, img_type, flag, transfroms=None) :
        self.xray_data_set = XrayDataSet(root, annotation, class_name, img_type, flag, transfroms)
        self.batch_size = batch_size
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.xray_data_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            collate_fn=collate_fn,)

    def get_data_loader(self):
        return self.data_loader

