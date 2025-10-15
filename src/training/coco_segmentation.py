import json
import os
from pathlib import Path
from typing import Any, Tuple

import torch
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import Resize, InterpolationMode
from torchvision.transforms import ToTensor

from src.training.data_transforms import RandomResizedCropWithPos


class CocoSegmentation(CocoDetection):
    n_categories = 91

    def __init__(self, patch_size, image_size=224, **kwargs) -> None:
        self.patch_size = patch_size
        self.mask_size = image_size // patch_size
        self.mask_resize = Resize(size=self.mask_size, interpolation=InterpolationMode.NEAREST)
        self.return_index_instead_of_target = True
        self.to_tensor = ToTensor()
        super().__init__(**kwargs)

        # Infer the location of the files
        self.real_root = str(Path(self.root).parents[1])
        self.phase = 'val' if 'val' in self.root.split('/')[-1] else 'train'
        self.seg_folder = f"annotations/stuff_annotations_trainval2017/annotations/stuff_{self.phase}2017_pixelmaps/"
        self.seg_folder = os.path.join(self.real_root, self.seg_folder)
        self.json_file = f"annotations/stuff_annotations_trainval2017/annotations/stuff_{self.phase}2017.json"
        self.json_file = os.path.join(self.real_root, self.json_file)

        # Load the .json
        with open(self.json_file) as f:
            self.ann_json = json.load(f)
            # all_cat = an_json['categories']

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        # Instantiate the thing mask
        channel_dict = {}
        w, h = image.size
        for t in target:
            channel_mask = self.coco.annToMask(t)
            channel_id = t['category_id'] - 1
            if channel_id in channel_dict:
                channel_dict[channel_id] += channel_mask
            else:
                channel_dict[channel_id] = channel_mask

        if len(channel_dict) > 0:
            masks = torch.stack([torch.tensor(channel_mask) for channel_mask in channel_dict.values()])
            channel_ids = torch.tensor(list(channel_dict.keys()), dtype=torch.int64)
            masks[masks > 1] = 1
        else:
            masks = torch.zeros([1, h, w])
            channel_ids = torch.tensor([0], dtype=torch.int64)

        # Transform the input
        if self.transforms is not None:
            image, masks = self.transforms((image, masks))

        # Pixels to patch
        masks = self.mask_resize(masks)
        masks = masks.to(torch.int64)
        h, w = masks[0].shape
        full_mask = torch.zeros([self.n_categories, h, w], dtype=torch.int64)
        full_mask[channel_ids] = masks
        # print(index, self.to_tensor(image).shape, full_mask.shape)
        # return torch.rand(3, 224, 224), torch.rand(91, 7, 7)
        # assert (type(self.to_tensor(image)) == torch.Tensor)
        # assert (type(full_mask) == torch.Tensor)
        return image, full_mask


if __name__ == '__main__':
    root_dir = '/mnt/mp600_2Tb/KU_LEUVEN/coco'
    patch_size = 32
    t = RandomResizedCropWithPos(size=224)
    keyword_args = {"root": os.path.join(root_dir, 'images/val2017'),
                    "annFile": os.path.join(root_dir, 'annotations/instances_val2017.json'),
                    "transform": None, "target_transform": None, "transforms": t}
    dataset = CocoSegmentation(patch_size, **keyword_args)
    img, mask = dataset.__getitem__(0)
    import matplotlib.pyplot as plt

    plt.imshow(img.permute(1, 2, 0))
    plt.show()

    print(mask.shape)

    for index, m in enumerate(mask):
        if m.sum() > 0:
            print(index)
            print(m)
            plt.imshow(m)
            plt.show()
