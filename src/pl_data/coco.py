import os
import cv2
from torchvision.datasets import CocoDetection

import src.pl_data.copy_paste
from src.pl_data.copy_paste import copy_paste_class
import skimage.io as io
import albumentations as A
from src.common.download import download_data
from omegaconf import ValueNode
from albumentations.pytorch import ToTensor
import torch

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
        root: ValueNode,
        path: ValueNode,
        image_size: ValueNode,
        image_exist: ValueNode,
        ann_file,
        transforms: ValueNode, max_size: ValueNode = None, augment_size: ValueNode = 1, filter_classes: ValueNode=['person']

    ):
        # print(transforms)
        # transforms = A.Compose(transforms.values(), bbox_params=A.BboxParams(format="coco", min_visibility=0.05))
        transforms = A.Compose([
            A.RandomScale(scale_limit=(-0.9, 1), p=1),  # LargeScaleJitter from scale of 0.1 to 2
            # A.PadIfNeeded(408, 408, border_mode=0), #pads with image in the center, not the top left like the paper
            A.RandomSizedBBoxSafeCrop(416, 416),
            src.pl_data.copy_paste.CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1.), # pct_objects_paste is a guess
            ToTensor()
        ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
        )
        anno_path = os.path.join(root, ann_file)
        if not os.path.exists(anno_path):
            download_data("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", root)
        super(CocoDetectionCP, self).__init__(
            root, anno_path, None, None, transforms
        )
        self.exist = image_exist
        # filter images without detection annotations
        ids = []
        img_ids = self.coco.getImgIds()
        person_id = self.coco.getCatIds(catNms=filter_classes)
        person_img_ids = self.coco.getImgIds(catIds=person_id)
        img_ids = list(set(img_ids) - set(person_img_ids))
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids
        person_ids = []

        for img_id in person_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                person_ids.append(img_id)
        self.person_ids = person_ids
        self.person_id = person_id

    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        img = self.coco.loadImgs(img_id)[0]
        if self.exist:
            path = self.coco.loadImgs(img_id)[0]['file_name']
            image = io.imread(os.path.join(self.root, path))
        else:
            image = io.imread(img['coco_url'])
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

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

    def load_person(self, index):
        img_id = self.person_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        img = self.coco.loadImgs(img_id)[0]
        if self.exist:
            path = self.coco.loadImgs(img_id)[0]['file_name']
            image = io.imread(os.path.join(self.root, path))
        else:
            image = io.imread(img['coco_url'])
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #convert all of the target segmentations to masks
        #bboxes are expected to be (y1, x1, y2, x2, category_id)
        masks = []
        bboxes = []
        for ix, obj in enumerate(target):
            masks.append(self.coco.annToMask(obj))
            if obj['category_id'] in self.person_id:
                bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])

        # person_filter = filter(lambda b: b[4] == 1, bboxes)
        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes[0]
        }
        return self.transforms(**output)

    # def __len__(self):
    #     return len(self.ids)
    #
    # def __repr__(self) -> str:
    #     return f"CocoDataset({self.name}, {self.path})"
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
    # def __len__(self):
    #     return len(self.ids)
    #
    # def __repr__(self) -> str:
    #     return f"CocoDataset({self.name}, {self.path})"
