import random

import torch
import torchvision.transforms.functional as F_viz
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms.transforms import RandomResizedCrop, Compose, RandomHorizontalFlip


class CenterCropWithPos(RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_size = 10000
        x = torch.arange(max_size).repeat(max_size, 1)[None, :]
        CenterCropWithPos._pos = torch.cat((x, x.permute(0, 2, 1)), dim=0).float()
        self.transform = transforms.CenterCrop(224)

    def forward(self, img):
        # PIL convention
        w_pil, h_pil = img.size
        pos = RandomResizedCropWithPos._pos[:, :h_pil, :w_pil]
        out = self.transform(img)
        out_pos = self.transform(pos)
        return out, out_pos


class RandomResizedCropWithPos(RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_size = 10000
        x = torch.arange(max_size).repeat(max_size, 1)[None, :]
        RandomResizedCropWithPos._pos = torch.cat((x, x.permute(0, 2, 1)), dim=0).float()
        self.scale = (0.5, 1.0)

    def forward(self, img):
        mask = None
        if isinstance(img, tuple):
            img, mask = img
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        # PIL convention
        w_pil, h_pil = img.size
        pos = RandomResizedCropWithPos._pos[:, :h_pil, :w_pil]
        out = F_viz.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        out_pos = F_viz.resized_crop(pos, i, j, h, w, self.size, self.interpolation)
        if mask is not None:
            mask = F_viz.resized_crop(mask, i, j, h, w, self.size, transforms.InterpolationMode.NEAREST)
        return out, out_pos, mask


class MyCompose(Compose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        flip_bool = 0
        pos = None
        mask = None
        for t in self.transforms:
            if type(t) == RandomResizedCropWithPos or type(t) == CenterCropWithPos:
                if isinstance(img, tuple):
                    img, pos, mask = t(img)
                else:
                    img, pos = t(img)
            elif type(t) == MyComposeInner:
                img, flip_bool = t(img)
            else:
                img = t(img)
        if flip_bool == 1:
            return img, F_viz.hflip(pos), F_viz.hflip(mask) if mask is not None else None
        return img, pos, mask


class MyComposeInner(Compose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        flip_bool = 0
        for t in self.transforms:
            if type(t) == RandomHorizontalFlipWithFlipBool:
                img, flip_bool = t(img)
            else:
                img = t(img)
        return img, flip_bool


class RandomHorizontalFlipWithFlipBool(RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        if torch.rand(1) < self.p:
            return F_viz.hflip(img), 1
        return img, 0


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationCrOC(object):
    def __init__(self, global_crops_scale, mean, std, image_size):
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # Spatial transformation
        self.spatial_transfo = MyCompose([
            RandomResizedCropWithPos(image_size, scale=global_crops_scale,
                                     interpolation=transforms.InterpolationMode.BICUBIC),
            MyComposeInner([RandomHorizontalFlipWithFlipBool(p=0.5)]),
        ])

        # Color transformations
        self.color_transfo1 = transforms.Compose([
            color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        self.color_transfo2 = transforms.Compose([
            color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])

    def __call__(self, image):
        mask = None
        if isinstance(image, tuple):
            image, mask = image

        # Apply the spatial transformations
        view_1_, pos_1, mask_1 = self.spatial_transfo((image, mask))
        view_2_, pos_2, mask_2 = self.spatial_transfo((image, mask))

        view_1, view_2 = self.color_transfo1(view_1_), self.color_transfo2(view_2_)
        crops = [view_1, view_2]
        crops_pos = [pos_1, pos_2]
        if mask is None:
            return crops, crops_pos
        crops_mask = [mask_1, mask_2]
        return crops, crops_pos, crops_mask


class DataAugmentationCOCO(object):
    # Hard code the hyperparameters
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    image_size = 224
    scale = (0.9, 1.0)

    def __init__(self, ):
        # Spatial transformation
        self.spatial_transfo = MyCompose([
            RandomResizedCropWithPos(self.image_size, scale=self.scale,
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        ])

        # Normalize
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def __call__(self, image):
        mask = None
        if isinstance(image, tuple):
            image, mask = image

        # Apply the spatial transformations
        image, _, mask = self.spatial_transfo((image, mask))

        # Normalize
        image = self.normalize(image)

        if mask is None:
            return image
        return image, mask
