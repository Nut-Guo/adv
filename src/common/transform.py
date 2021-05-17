import torch
import torch.functional as F


class Resize:
    """Rescales the image and target to given dimensions.

    Args:
        output_size (tuple or int): Desired output size. If tuple (height, width), the output is
            matched to ``output_size``. If int, the smaller of the image edges is matched to
            ``output_size``, keeping the aspect ratio the same.
    """

    def __init__(self, output_size: tuple) -> None:
        self.output_size = output_size

    def __call__(self, image, target):
        width, height = image.size
        original_size = torch.tensor([height, width])
        resize_ratio = torch.tensor(self.output_size) / original_size
        image = F.resize(image, self.output_size)
        scale = torch.tensor(
            [
                resize_ratio[1],  # y
                resize_ratio[0],  # x
                resize_ratio[1],  # y
                resize_ratio[0]  # x
            ],
            device=target['boxes'].device
        )
        target['boxes'] = target['boxes'] * scale
        return image, target
