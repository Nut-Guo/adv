import os
import cv2
from torchvision.datasets import CocoDetection

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


class Background(CocoDetection):
    def __init__(
        self,
        root: ValueNode='/content/adv/data/train2017',
        image_exist=True,
        ann_file='annotations/instances_train2017.json',
    ):
        transforms = A.Compose([
            A.RandomScale(scale_limit=(-0.9, 1), p=1),  # LargeScaleJitter from scale of 0.1 to 2
            A.RandomSizedBBoxSafeCrop(416, 416),
            ToTensor()
        ])
        anno_path = os.path.join(root, ann_file)
        if not os.path.exists(anno_path):
            download_data("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", root)
        super(Background, self).__init__(
            root, anno_path, None, None, transforms
        )
        self.exist = image_exist
        img_ids = self.coco.getImgIds()
        person_id = self.coco.getCatIds(catNms=['person'])
        person_img_ids = self.coco.getImgIds(catIds=person_id)
        img_ids = list(set(img_ids) - set(person_img_ids))
        self.ids = img_ids

    def load_background(self, index):
        img_id = self.ids[index]
        img = self.coco.loadImgs(img_id)[0]
        if self.exist:
            path = self.coco.loadImgs(img_id)[0]['file_name']
            image = io.imread(os.path.join(self.root, path))
        else:
            image = io.imread(img['coco_url'])
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return self.transforms(image)

    def __len__(self):
        return len(self.ids)

    def __repr__(self) -> str:
        return f"Background({self.name}, {self.path})"
