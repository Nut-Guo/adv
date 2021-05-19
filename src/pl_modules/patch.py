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

    def __init__(self, image_size, patch_size, portion=1, degrees=None, translate=None, scale=None, brightness=None, contrast=None,
                 saturation=None, hue=None):
        super(PatchTransformer, self).__init__()
        self.image_size = image_size
        self.portion = portion
        self.pad_size = (image_size - patch_size) // 2
        self.transforms = transforms.Compose([
            # transforms.Pad(
            #     self.pad_size
            # ),
            transforms.RandomAffine(
                degrees=degrees,
                translate=translate,
                scale=scale,
                # shear=[-1, 1, -1, 1]
            ),
            # transforms.ColorJitter(
            #     brightness=brightness,
            #     contrast=contrast,
            #     saturation=saturation,
            #     hue=hue
            # ),
        ])
        '''
        kernel = torch.cuda.FloatTensor([[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],                                                                                    
                                         [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],                                                                                    
                                         [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]])
        self.kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,3,-1,-1)
        '''

    @staticmethod
    def generate_tensor(adv_batch, batch_size, min, max):
        result = torch.tensor(batch_size, dtype=torch.float).uniform_(min, max)
        result = result.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        result = result.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        return result

    def placeinto_box(self, patch, box, base):
        # box[2] = min(self.image_size, box[2])
        # box[3] = min(self.image_size, box[3])
        box = [min(self.image_size, int(p)) for p in box]
        size = int(min(box[2] - box[0], box[3]-box[1]) // 2) * 2 * self.portion
        trans = transforms.Resize((size, size))
        patch = trans(patch)
        midx = (box[2] - box[0])//2
        midy = (box[3] - box[1])//2
        x1 = midx - size//2
        x2 = midx + size//2
        y1 = midy - size//2
        y2 = midy + size//2
        print(x1, x2, y1, y2)
        base[:, x1:x2, y1:y2] = patch
        return base

    def forward(self, adv_patch, ground_truth):
        """
        Transform the adv patch according to the bounding boxes in ground truth
        Args:
            adv_patch: patch to be applied
            ground_truth: bounding boxes from yolo detection.

        Returns:
            Tensors of the same shape as images with patch in the middle of the targets
        """
        adv_batch = []
        for boxes in ground_truth['boxes']:
            base = torch.zeros((3, self.image_size, self.image_size), device = 'cuda')
            for box in boxes:
                base = self.placeinto_box(adv_patch, box, base)
            adv_batch.append(base)
        adv_batch = self.transforms(torch.stack(adv_batch))
        return adv_batch
        # return self.transforms(adv_patch)
    # def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
    #     # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
    #     adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
    #     # Determine size of padding
    #     pad = (img_size - adv_patch.size(-1)) / 2
    #     # Make a batch of patches
    #     adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
    #     adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
    #     batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))
    #
    #     # Contrast, brightness and noise transforms
    #
    #     # Create random contrast tensor
    #     contrast = self.generate_tensor(adv_batch, batch_size, self.min_contrast, self.max_contrast)
    #
    #     # Create random brightness tensor
    #     brightness = self.generate_tensor(adv_batch, batch_size, self.min_brightness, self.max_brightness)
    #
    #     # Create random noise tensor
    #     noise = torch.tensor(adv_batch.size(), dtype=torch.float).uniform_(-1, 1) * self.noise_factor
    #
    #     # Apply contrast/brightness/noise, clamp
    #     adv_batch = adv_batch * contrast + brightness + noise
    #
    #     adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
    #
    #     # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
    #     cls_ids = torch.narrow(lab_batch, 2, 0, 1)
    #     cls_mask = cls_ids.expand(-1, -1, 3)
    #     cls_mask = cls_mask.unsqueeze(-1)
    #     cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
    #     cls_mask = cls_mask.unsqueeze(-1)
    #     cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
    #     msk_batch = torch.ones_like(cls_mask) - cls_mask
    #
    #     # Pad patch and mask to image dimensions
    #     mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
    #     adv_batch = mypad(adv_batch)
    #     msk_batch = mypad(msk_batch)
    #
    #     # Rotation and rescaling transforms
    #     anglesize = (lab_batch.size(0) * lab_batch.size(1))
    #     if do_rotate:
    #         angle = torch.tensor(anglesize, dtype=torch.float).uniform_(self.minangle, self.maxangle)
    #     else:
    #         angle = torch.zeros_like(anglesize, dtype=torch.float)
    #
    #     # Resizes and rotates
    #     current_patch_size = adv_patch.size(-1)
    #     lab_batch_scaled = torch.tensor(lab_batch.size(), dtype=torch.float).fill_(0)
    #     lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
    #     lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
    #     lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
    #     lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
    #     target_size = torch.sqrt(
    #         ((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
    #     target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
    #     target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
    #     targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
    #     targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
    #     if rand_loc:
    #         off_x = targetoff_x * (torch.tensor(targetoff_x.size(), dtype=torch.float).uniform_(-0.4, 0.4))
    #         target_x = target_x + off_x
    #         off_y = targetoff_y * (torch.tensor(targetoff_y.size(), dtype=torch.float).uniform_(-0.4, 0.4))
    #         target_y = target_y + off_y
    #     target_y = target_y - 0.05
    #     scale = target_size / current_patch_size
    #     scale = scale.view(anglesize)
    #
    #     s = adv_batch.size()
    #     adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
    #     msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])
    #
    #     tx = (-target_x + 0.5) * 2
    #     ty = (-target_y + 0.5) * 2
    #     sin = torch.sin(angle)
    #     cos = torch.cos(angle)
    #
    #     # Theta = rotation,rescale matrix
    #     theta = torch.tensor(anglesize, 2, 3, dtype=torch.float).fill_(0)
    #     theta[:, 0, 0] = cos / scale
    #     theta[:, 0, 1] = sin / scale
    #     theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
    #     theta[:, 1, 0] = -sin / scale
    #     theta[:, 1, 1] = cos / scale
    #     theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
    #
    #     b_sh = adv_batch.shape
    #     grid = F.affine_grid(theta, adv_batch.shape)
    #
    #     adv_batch_t = F.grid_sample(adv_batch, grid)
    #     msk_batch_t = F.grid_sample(msk_batch, grid)
    #
    #     '''
    #     # Theta2 = translation matrix
    #     theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
    #     theta2[:, 0, 0] = 1
    #     theta2[:, 0, 1] = 0
    #     theta2[:, 0, 2] = (-target_x + 0.5) * 2
    #     theta2[:, 1, 0] = 0
    #     theta2[:, 1, 1] = 1
    #     theta2[:, 1, 2] = (-target_y + 0.5) * 2
    #
    #     grid2 = F.affine_grid(theta2, adv_batch.shape)
    #     adv_batch_t = F.grid_sample(adv_batch_t, grid2)
    #     msk_batch_t = F.grid_sample(msk_batch_t, grid2)
    #
    #     '''
    #     adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
    #     msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])
    #
    #     adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
    #     # img = msk_batch_t[0, 0, :, :, :].detach().cpu()
    #     # img = transforms.ToPILImage()(img)
    #     # img.show()
    #     # exit()
    #
    #     return adv_batch_t * msk_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    @staticmethod
    def forward(img_batch, adv_batch):
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
