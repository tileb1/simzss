import json
import os
from collections import OrderedDict

import torch
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2


class COCOSegmentationEZ(CocoDetection):
    def __init__(self, patch_size, transforms, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.seg_transforms = transforms

    def __getitem__(self, index):
        # Get the path of the image
        id = self.ids[index]
        path = self.coco.loadImgs(id)[0]["file_name"]
        path = os.path.join(self.root, path)

        # Get the image and targets
        image_pil, annotations = super().__getitem__(index)

        # Instantiate the thing mask
        channel_dict = {}
        w, h = image_pil.size
        for annotation in annotations:
            channel_mask = self.coco.annToMask(annotation)
            channel_id = annotation['category_id']
            if channel_id in channel_dict:
                channel_dict[channel_id] += channel_mask
            else:
                channel_dict[channel_id] = channel_mask

        # Stack
        if len(channel_dict) > 0:
            masks = torch.stack([torch.tensor(channel_mask) for channel_mask in channel_dict.values()])
            labels = torch.tensor(list(channel_dict.keys()), dtype=torch.int64)
            masks[masks > 1] = 1
        else:
            masks = torch.zeros([1, h, w])
            labels = torch.tensor([0], dtype=torch.int64)

        # Transform
        masks = tv_tensors.Mask(masks)
        image, masks = self.seg_transforms(image_pil, masks)

        _, ht, wt = image.shape
        image = image[:, :ht - ht % self.patch_size, :wt - wt % self.patch_size]
        _, hm, wm = masks.shape
        assert (hm == ht)
        assert (wm == wt)
        masks = masks[:, :ht - ht % self.patch_size, :wt - wt % self.patch_size]

        # Down-sample the masks to the patch resolution
        masks = torch.nn.functional.interpolate(masks[None, :], size=(ht // self.patch_size, wt // self.patch_size),
                                                mode='nearest').squeeze(0)

        # Drop empty channels
        keep_channels = masks.sum(dim=(-1, -2)) > 0
        masks = masks[keep_channels]
        labels = labels[keep_channels]

        # Handle empty masks
        if masks.sum() == 0.:
            masks = None
            labels = None
        return image_pil, image, masks, labels, path


class COCOSegmentationEZMapped80(COCOSegmentationEZ):
    def __init__(self, patch_size, transforms, **kwargs):
        coco_things_classes_path = kwargs['coco_things_classes_path']
        del kwargs['coco_things_classes_path']
        super().__init__(patch_size, transforms, **kwargs)

        # Get COCO things mapping
        with open(coco_things_classes_path, 'r') as f:
            self.id_to_class_coco = json.load(f, object_pairs_hook=OrderedDict)
            self.label_to_index = {int(k): i for i, k in enumerate(self.id_to_class_coco.keys())}

    def __getitem__(self, index):
        _, image, masks, labels, _ = super().__getitem__(index)

        if labels is not None:
            labels.apply_(lambda x: self.label_to_index[x])
            onehotmasks = torch.zeros(len(self.label_to_index), *masks.shape[1:], dtype=torch.uint8)
            onehotmasks[labels] = masks
        else:
            onehotmasks = None
        return image, onehotmasks


if __name__ == '__main__':
    transforms = v2.Compose([
        v2.Resize(size=224, interpolation=v2.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        v2.CenterCrop(size=(224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    coco_dir = '/media/thomas/Elements/cv_datasets/coco'
    train_dataset = COCOSegmentationEZ(
        transforms=transforms,
        root=os.path.join(coco_dir, 'images/train2017'),
        annFile=os.path.join(coco_dir, 'annotations/instances_train2017.json'),
    )

    train_dataset.__getitem__(0)
