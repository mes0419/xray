import os

import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data


class XrayDataSet(data.Dataset):
    def __init__(self, root, annotation, class_name=None, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.class_name = class_name
        self.transforms = transforms

        if self.class_name is not None:
            class_id = sorted(self.coco.getCatIds(catNms=self.class_name))
            self.ids = self.make_id_list(class_id)

        else:
            self.ids = list(sorted(self.coco.imgs.keys()))

        self.ids = list(set(self.ids))

    def __getitem__(self, item):
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

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        class_id = torch.as_tensor(class_id)

        image_id = coco_annotation[0]["image_id"]
        file_name = img_info[0]["file_name"]

        # Tensorise
        image_id = torch.tensor([image_id])
        # file_name = torch.tensor([file_name])

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Annotation is in dictionary format
        xray_annotation = {}
        xray_annotation["boxes"] = boxes
        xray_annotation["class_id"] = class_id
        xray_annotation["image_id"] = image_id
        # xray_annotation["file_name"] = file_name

        if self.transforms is not None:
            img = self.transforms(img)

        return img, xray_annotation

    def make_id_list(self, class_id):
        # 공유폴더에 이미지가 존재하는것만 index를 만들어줌
        ids = []
        for id in class_id:
            find_id = self.coco.getImgIds(catIds=[id])
            for i in find_id:
                file = self.coco.imgs[i]['path'].split('\\', maxsplit=7)[-1].replace('\\', '/')
                if os.path.isfile(os.path.join(self.root, file)):
                    ids.append(i)
                else:
                    print("[DEBUG] Check Empty file :", file)
        return ids

    def __len__(self):
        return len(self.ids)


def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


class XrayDataLoader():
    def __init__(self, root, annotation, batch_size, class_name=None, transfroms=None):
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