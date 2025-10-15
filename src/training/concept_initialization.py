import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.transforms import v2
from tqdm import tqdm

import open_clip
from dino_utils import init_distributed_mode


def get_parser():
    parser = argparse.ArgumentParser('Initialization and evalutation on COCO.')
    parser.add_argument('--coco_dir', default='/media/thomas/Elements/cv_datasets/coco', type=str, help='path to coco')
    parser.add_argument('--batch_size_per_gpu', default=16, type=int, help='Per-GPU batch-size')
    parser.add_argument('--image_size', default=224, type=int, help='Size of the images after transforms')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def initialize_clip(classes, savepath='class_names/normalized_in21k_vitb16_laion2b_s34b_b88k.pth'):
    # Get the model
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    clip_model = clip_model.cuda()
    clip_model = clip_model.eval()

    # Get coco classes
    np_classes = np.array(classes)

    # Get the tokenizer
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    if not os.path.exists(savepath):
        # Tokenize
        tokens = tokenizer(classes).cuda()

        # Feed the coco classes to the text encoder
        chunk_size = 128
        n_chunks = tokens.shape[0] // chunk_size + 1
        class_embeddings = []
        for chunk in torch.chunk(tokens, chunks=n_chunks, dim=0):
            with torch.no_grad():
                class_embeddings.append(clip_model.encode_text(chunk))
        class_embeddings = torch.cat(class_embeddings, dim=0)
        class_embeddings = torch.nn.functional.normalize(class_embeddings, dim=-1, p=2)
        torch.save(class_embeddings, savepath)
    else:
        class_embeddings = torch.load(savepath).cuda()

    # Make the ViT return the spatial tokens
    clip_model.visual.output_tokens = True

    # Get the last normalization out of the model
    ln_post = clip_model.visual.ln_post
    clip_model.visual.ln_post = torch.nn.Identity()

    # Get the projections out of the ViT
    image_to_text = clip_model.visual.proj
    clip_model.proj = None

    # Store in dict for convenience
    clip_stuff = {
        'clip_model': clip_model,
        'ln_post': ln_post,
        'image_to_text': image_to_text,
        'classes': np_classes,
        'class_embeddings': class_embeddings,
    }
    return clip_stuff


def coco_collate(batch):
    images, _, _, paths = list(zip(*batch))
    images = torch.stack(images)
    return images, paths


def get_data_bboxes(args, transforms):
    # Datasets
    train_dataset = CocoDetection(
        root=os.path.join(args.coco_dir, 'images/train2017'),
        annFile=os.path.join(args.coco_dir, 'annotations/instances_train2017.json'),
        transforms=transforms,
    )
    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    val_dataset = CocoDetection(
        root=os.path.join(args.coco_dir, 'images/val2017'),
        annFile=os.path.join(args.coco_dir, 'annotations/instances_val2017.json'),
        transforms=transforms,
    )
    val_dataset = wrap_dataset_for_transforms_v2(val_dataset)
    return train_dataset, val_dataset


def get_data(args):
    # Set the transforms
    transforms = v2.Compose([
        v2.Resize(size=args.image_size, interpolation=v2.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        v2.CenterCrop(size=(args.image_size, args.image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Get the datasets
    train_dataset, val_dataset = get_data_bboxes(args, transforms)

    # Dataloaders
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=False)
    train_loader = DataLoader(
        sampler=train_sampler,
        dataset=train_dataset,
        batch_size=args.batch_size_per_gpu,
        collate_fn=coco_collate,
    )
    val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        sampler=val_sampler,
        dataset=val_dataset,
        batch_size=args.batch_size_per_gpu,
        collate_fn=coco_collate,
    )
    return train_loader, val_loader


class ModelWrapper(torch.nn.Module):
    def __init__(self, clip_stuff):
        super().__init__()
        self.model = clip_stuff['clip_model']
        self.ln_post = clip_stuff['ln_post']
        self.image_to_text = clip_stuff['image_to_text']

    def forward(self, images):
        _, visual_tokens = self.model.visual(images)

        # Project to the text space
        visual_tokens = self.ln_post(visual_tokens)
        visual_tokens = visual_tokens @ self.image_to_text
        visual_tokens = torch.nn.functional.normalize(visual_tokens, dim=-1, p=2)
        return visual_tokens


def image_to_concepts(loader, model, savepath, concepts):
    for images in tqdm(loader):
        # Move to GPU
        images, paths = images.cuda()


def main(args):
    # Fully initialize distributed device environment
    init_distributed_mode(args)

    # Get IN21K classes
    with open('class_names/imagenet21k.txt', 'r') as f:
        in21k_classes = [l.strip() for l in f.readlines()]

    # Initialize CLIP
    clip_stuff = initialize_clip(in21k_classes)

    # Get the dataloaders
    train_loader, val_loader = get_data(args)

    # Wrap the model
    model = ModelWrapper(clip_stuff).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Map images to concepts
    image_to_concepts(val_loader, model, )


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
