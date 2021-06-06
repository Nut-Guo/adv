import os
import cv2
from torchvision.datasets import CocoDetection

import src.pl_data.copy_paste
from src.pl_data.copy_paste import copy_paste_class
import skimage.io as io
import albumentations as A
from src.pl_modules.yolo import NAME2ID
from src.common.download import download_data
from omegaconf import ValueNode
from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from src.common.utils import PROJECT_ROOT
from albumentations.pytorch import ToTensor
import torch
import cv2

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False

@copy_paste_class
class CocoDetectionCP(CocoDetection):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        image_size: ValueNode,
        ann_file,
        transforms: ValueNode, max_size: ValueNode = None, augment_size: ValueNode = 1, filter_classes: ValueNode=['person']

    ):
        # print(transforms)
        # transforms = A.Compose(transforms.values())
        transforms = A.Compose([
            A.RandomScale(scale_limit=(-0.9, 1), p=1),  # LargeScaleJitter from scale of 0.1 to 2
            # A.PadIfNeeded(408, 408, border_mode=0), #pads with image in the center, not the top left like the paper
            A.RandomSizedBBoxSafeCrop(408, 408),
            src.pl_data.copy_paste.CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1.)  # pct_objects_paste is a guess
        ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
        )
        ann_file = os.path.join(path, ann_file)
        if not os.path.exists(ann_file):
            download_data("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", path)
        super(CocoDetectionCP, self).__init__(
            path, ann_file, None, None, transforms
        )

        # filter images without detection annotations
        ids = []
        # catIds = self.coco.getCatIds(catNms=filter_classes)
        # self.ids = self.coco.getImgIds(catIds=catIds)
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids
        self.filter_classes = self.coco.getCatIds(catNms=filter_classes)

    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        img = self.coco.loadImgs(img_id)[0]
        image = io.imread(img['coco_url'])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #convert all of the target segmentations to masks
        #bboxes are expected to be (y1, x1, y2, x2, category_id)
        masks = []
        bboxes = []
        for ix, obj in enumerate(target):
            masks.append(self.coco.annToMask(obj))
            # if obj['category_id'] in self.filter_classes:
            bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])

        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }
        
        return self.transforms(**output)

# class CocoDataset(object):
#     def __init__(self, name: ValueNode, path: ValueNode, image_size: ValueNode, transforms: ValueNode,
#                  max_size: ValueNode = None, augment_size: ValueNode = 1, filter_classes: ValueNode=['person'], **kwargs):
#         super().__init__()
#         self.path = path
#         self.name = name
#         self.augment_size = augment_size
#         self.image_size = image_size
#         self.coco = COCO(path)
#         ids = []
#         catIds = self.coco.getCatIds(catNms=filter_classes)
#         self.ids = self.coco.getImgIds(catIds=catIds)
#         for img_id in self.ids:
#             ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
#             anno = self.coco.loadAnns(ann_ids)
#             if has_valid_annotation(anno):
#                 ids.append(img_id)
#         self.ids = ids
#         if max_size:
#             self.ids = self.ids[:max_size]
#         self.transforms = A.Compose(transforms + [
#             ToTensor()
#         ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05))
#
#     def __getitem__(self, idx):
#         img_id = self.ids[idx]
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         target = self.coco.loadAnns(ann_ids)
#
#         img = self.coco.loadImgs(img_id)[0]
#         image = io.imread(img['coco_url'])
#         masks = []
#         bboxes = []
#         for ix, obj in enumerate(target):
#             masks.append(self.coco.annToMask(obj))
#             bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])
#
#         # pack outputs into a dict
#         output = {
#             'image': image,
#             'masks': masks,
#             'bboxes': bboxes
#         }
#         result = self.transforms(**output)
#         return {
#             "image": torch.stack(result['image']),
#             "bboxes": torch.stack(result['bboxes']),
#         }
#
#     def __len__(self):
#         return len(self.ids)
#
#     def __repr__(self) -> str:
#         return f"CocoDataset({self.name}, {self.path})"
