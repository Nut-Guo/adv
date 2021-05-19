import os
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor
from torch import nn
from torchvision.ops import nms
from src.common.utils import MODEL_PATH
from src.common.download import download_data
from pl_bolts.models.detection import YOLO, YOLOConfiguration

CONFIG3_BASE_URL = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/'
WEIGHTS3_BASE_URL = 'https://pjreddie.com/media/files/'
CONFIG4_BASE_URL = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/'
WEIGHTS4_BASE_URL = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/'
NAMES_URL = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
NAMES_PATH = os.path.join(MODEL_PATH, 'coco.names')


def prepare_files(name):
    if name.startswith('yolov3'):
        config_base_url = CONFIG3_BASE_URL
        weights_base_url = WEIGHTS3_BASE_URL
    else:
        config_base_url = CONFIG4_BASE_URL
        weights_base_url = WEIGHTS4_BASE_URL
    # cfg_path = os.path.join(MODEL_PATH, config)
    # weights_path = cfg_path.replace('cfg', 'weights')
    cfg_url = config_base_url + name + '.cfg'
    weights_url = weights_base_url + name + '.weights'
    print("Downloading {}".format(cfg_url))
    download_data(cfg_url, MODEL_PATH)
    print("Downloading {}".format(weights_url))
    download_data(weights_url, MODEL_PATH)


def get_yolo(name):
    path_name = os.path.join(MODEL_PATH, name)
    config_path = path_name + '.cfg'
    weights_path = path_name + '.weights'
    if not os.path.exists(config_path):
        prepare_files(name)
    config = YOLOConfiguration(config_path)
    model = YOLO(config.get_network())
    with open(weights_path, 'rb') as f:
        model.load_darknet_weights(f)
    return model, config


def get_names():
    download_data(NAMES_URL, MODEL_PATH)
    with open(NAMES_PATH) as f:
        names = f.read().split('\n')
    names = list(filter(None, names))
    names2id = {name: i for i, name in enumerate(names)}
    return names, names2id


NAMES, NAME2ID = get_names()


def split_detections(detections: Tensor) -> Dict[str, Tensor]:
    """
    Splits the detection tensor returned by a forward pass into a dictionary.

    The fields of the dictionary are as follows:
        - boxes (``Tensor[batch_size, N, 4]``): detected bounding box `(x1, y1, x2, y2)` coordinates
        - scores (``Tensor[batch_size, N]``): detection confidences
        - classprobs (``Tensor[batch_size, N]``): probabilities of the best classes
        - labels (``Int64Tensor[batch_size, N]``): the predicted labels for each image

    Args:
        detections: A tensor of detected bounding boxes and their attributes.

    Returns:
        A dictionary of detection results.
    """
    boxes = detections[..., :4]
    scores = detections[..., 4]
    classprobs = detections[..., 5:]
    classprobs, labels = torch.max(classprobs, -1)
    return {'boxes': boxes, 'scores': scores, 'classprobs': classprobs, 'labels': labels}


def filter_detections(detections: Dict[str, Tensor], confidence_threshold: float = 0.2,
                      nms_threshold: float = 0.45, max_predictions_per_image: int = -1,
                      target_class: str = None) -> Dict[str, List[Tensor]]:
    """
    Filters detections based on confidence threshold. Then for every class performs non-maximum
    suppression (NMS). NMS iterates the bounding boxes that predict this class in descending
    order of confidence score, and removes lower scoring boxes that have an IoU greater than
    the NMS threshold with a higher scoring box. Finally the detections are sorted by descending
    confidence and possible truncated to the maximum number of predictions.

    Args:
        detections: All detections. A dictionary of tensors, each containing the predictions
            from all images.
        confidence_threshold: Confidence threshold for filtering low confidence detections.
        nms_threshold: NMS threshold to select boxes.
        max_predictions_per_image: Max output predictions.
        target_class: Filter out only detections for target classes.

    Returns:
        Filtered detections. A dictionary of lists, each containing a tensor per image.
    """
    boxes = detections['boxes']
    scores = detections['scores']
    classprobs = detections['classprobs']
    labels = detections['labels']

    out_boxes = []
    out_scores = []
    out_classprobs = []
    out_labels = []

    for img_boxes, img_scores, img_classprobs, img_labels in zip(boxes, scores, classprobs, labels):
        # Select detections with high confidence score.
        selected = img_scores > confidence_threshold
        if target_class:
            selected = selected.logical_and(img_labels == NAME2ID[target_class])
        img_boxes = img_boxes[selected]
        img_scores = img_scores[selected]
        img_classprobs = img_classprobs[selected]
        img_labels = img_labels[selected]

        img_out_boxes = boxes.new_zeros((0, 4))
        img_out_scores = scores.new_zeros(0)
        img_out_classprobs = classprobs.new_zeros(0)
        img_out_labels = labels.new_zeros(0)

        # Iterate through the unique object classes detected in the image and perform non-maximum
        # suppression for the objects of the class in question.
        for cls_label in labels.unique():
            selected = img_labels == cls_label
            cls_boxes = img_boxes[selected]
            cls_scores = img_scores[selected]
            cls_classprobs = img_classprobs[selected]
            cls_labels = img_labels[selected]

            selected = nms(cls_boxes, cls_scores, nms_threshold)
            img_out_boxes = torch.cat((img_out_boxes, cls_boxes[selected]))
            img_out_scores = torch.cat((img_out_scores, cls_scores[selected]))
            img_out_classprobs = torch.cat((img_out_classprobs, cls_classprobs[selected]))
            img_out_labels = torch.cat((img_out_labels, cls_labels[selected]))

        # Sort by descending confidence and limit the maximum number of predictions.
        indices = torch.argsort(img_out_scores, descending=True)
        if max_predictions_per_image >= 0:
            indices = indices[:max_predictions_per_image]
        out_boxes.append(img_out_boxes[indices])
        out_scores.append(img_out_scores[indices])
        out_classprobs.append(img_out_classprobs[indices])
        out_labels.append(img_out_labels[indices])

    return {'boxes': out_boxes, 'scores': out_scores, 'classprobs': out_classprobs, 'labels': out_labels}


class PredExtractor(nn.Module):
    """ProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, target_class):
        super(PredExtractor, self).__init__()
        self.target_class = target_class

    def forward(self, detection):
        detections = split_detections(detection)
        detections = filter_detections(detections, target_class=self.target_class)
        return detections


def main():
    # get_names()
    prepare_files('yolov4-tiny')
    model, config = get_yolo('yolov4-tiny')
    from src.pl_data.dataset import PersonDataset
    path = '/content/adv/data/LIP/Images/train_images'
    names = list(sorted(os.listdir(path)))
    from torchvision import transforms
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((416, 416))])
    from PIL import Image
    with open("blacklist.txt", "a") as f:
        for name in names:
            img = Image.open(os.path.join(path, name))
            img = trans(img)
            result = model.infer(img)
            if not NAME2ID['person'] in result[2]:
                f.write(name + '\n')
                print(result[2])


if __name__ == "__main__":
    main()
