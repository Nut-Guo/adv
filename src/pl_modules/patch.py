from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import math
# from median_pool import MedianPool2d
import os
from PIL import Image


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001)
        # tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001)
        # tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1.sum() + tvcomp2.sum()
        return tv / torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self, image_size, patch_batch_size, patch_size, portion=1, patch_transforms=None):
        super(PatchTransformer, self).__init__()
        self.image_size = image_size
        self.portion = portion
        self.patch_size = patch_size
        self.patch_batch_size = patch_batch_size
        self.base = nn.Parameter(torch.zeros((3, self.image_size, self.image_size)))
        self.register_parameter(name='base', param=self.base)
        self.transforms = transforms.Compose(list(patch_transforms.values()))

    @staticmethod
    def generate_tensor(adv_batch, batch_size, min, max):
        result = torch.tensor(batch_size, dtype=torch.float).uniform_(min, max)
        result = result.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        result = result.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        return result

    def placeinto_box(self, patch, box, base):
        # box[2] = min(self.image_size, box[2])
        # box[3] = min(self.image_size, box[3])
        box = box.clamp(0, self.image_size)
        box = [int(p) for p in box]
        midx = (box[3] + box[1]) // 2
        midy = (box[2] + box[0]) // 2
        # if self.patch_size[1] > self.patch_size[0]:
        #     y2x = self.patch_size[1] / self.patch_size[0]
        #     xsize = int((min(box[2] - box[0], box[3]-box[1])) * self.portion)
        #     ysize = int(y2x * xsize)
        # else:
        #     x2y = self.patch_size[0] / self.patch_size[1]
        #     ysize = int((min(box[2] - box[0], box[3] - box[1])) * self.portion)
        #     xsize = int(x2y * ysize)
        y2x = self.patch_size[1] / self.patch_size[0]
        xsize = min(int(self.portion * (box[3] - box[1])), int(self.portion * (box[2] - box[0]) / y2x))
        ysize = int(y2x * xsize)
        # ysize = int(self.portion * (box[2] - box[0]))
        trans = transforms.Resize((xsize, ysize), interpolation=transforms.InterpolationMode.NEAREST)
        patch = trans(patch)
        x1 = midx - xsize//2
        y1 = midy - ysize//2
        x2 = x1 + xsize
        y2 = y1 + ysize
        mask = torch.where(patch != 0)
        base[:, x1:x2, y1:y2][mask] = patch[mask]
        return base

    def forward(self, adv_patch, boxes_batch):
        """
        Transform the adv patch according to the bounding boxes in ground truth
        Args:
            adv_patch: patch to be applied
            boxes_batch: bounding boxes from yolo detection.

        Returns:
            Tensors of the same shape as images with patch in the middle of the targets
        """
        adv_batch = []
        for boxes in boxes_batch:
            base = self.base.clone()
            # boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            # box_id = torch.argmax(boxes_area, 0)
            # for box in boxes:
            box = boxes[0]
            trans_adv_patch = self.transforms(adv_patch)
            base = self.placeinto_box(trans_adv_patch, box, base)
            adv_batch.append(base)
        adv_batch = torch.stack(adv_batch)
        return adv_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    @staticmethod
    def forward(img_batch, adv_batch):
        assert(img_batch.shape == adv_batch.shape)
        img_batch = torch.where((adv_batch == 0), img_batch, adv_batch)
        return img_batch


'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''


def main():
    patch = torch.rand((3, 100, 100))
    trans = PatchTransformer(416, 100, (-30, 30), (0.25, 0.25), (0.8, 1.2))
    app = PatchApplier()
    img = torch.ones((3, 3, 416, 416))
    patch = trans(patch)
    img = app(img, patch)
    import matplotlib.pyplot as plt
    plt.imshow(img[2].permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    main()
