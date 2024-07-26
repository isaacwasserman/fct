import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
import numpy as np
import os
import urllib.request
import tarfile
import shutil
import glob
from PIL import Image
from utils import DownloadProgressBar
from matplotlib.cm import tab20c

cub_palette = list(tab20c.colors)
cub_palette[0] = [0, 0, 0]
cub_palette = torch.tensor(cub_palette)


def create_image_grid(x, y_hat, y, width=None):
    if y.shape[1] > 1:
        y_hat_int = y_hat.argmax(dim=1, keepdim=False)
        y_int = y.argmax(dim=1, keepdim=False)
        y_hat_colorized = cub_palette[y_hat_int].permute(0, 3, 1, 2)
        y_colorized = cub_palette[y_int].permute(0, 3, 1, 2)
        lineups = torch.cat([x.cpu(), y_hat_colorized.cpu(), y_colorized.cpu()], dim=3)
    else:
        lineups = torch.cat([x.cpu(), y_hat.repeat(1, 3, 1, 1).cpu(), y.repeat(1, 3, 1, 1).cpu()], dim=3)
    grid = torchvision.utils.make_grid(lineups, nrow=2)
    return grid


perceptual_mean = [0.49139968, 0.48215841, 0.44653091]
perceptual_std = [0.24703223, 0.24348513, 0.26158784]


def transform(image, target, augment=True):
    longest_side = max(image.shape[0], image.shape[1])

    image = image.astype(np.float32)
    target = target.astype(np.float32)

    cropped_size = np.random.randint(int(0.6 * longest_side), longest_side)
    crop_top = np.random.randint(0, longest_side - cropped_size)
    crop_left = np.random.randint(0, longest_side - cropped_size)
    should_flip = np.random.randint(0, 2) == 1

    image = torch.tensor(image)
    target = torch.tensor(target)
    # Make square
    image = torchvision.transforms.functional.resize(image, (longest_side, longest_side))
    target = torchvision.transforms.functional.resize(
        target.unsqueeze(0), (longest_side, longest_side), interpolation=torchvision.transforms.InterpolationMode.NEAREST
    ).squeeze(0)
    # Crop
    image = transforms_v2.functional.resized_crop(
        image,
        crop_top,
        crop_left,
        cropped_size,
        cropped_size,
        (256, 256),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    target = transforms_v2.functional.resized_crop(
        target.unsqueeze(0),
        crop_top,
        crop_left,
        cropped_size,
        cropped_size,
        (256, 256),
        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
    ).squeeze(0)
    # Flip
    if should_flip:
        image = transforms_v2.functional.horizontal_flip(image)
        target = transforms_v2.functional.horizontal_flip(target)
    # Jitter
    image = transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(image)
    # Normalize
    image = torchvision.transforms.functional.normalize(image, perceptual_mean, perceptual_std)

    return image, target


train_transform = lambda image, target: transform(image, target, augment=True)
val_transform = lambda image, target: transform(image, target, augment=False)
test_transform = lambda image, target: transform(image, target, augment=False)


class CUB200SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, semantic=False):
        self.all_image_paths = np.array(list(glob.glob(f"{root}/images/*/*.jpg")))
        self.all_segmentation_paths = np.array(
            [
                "/".join(path.split("/")[:-3] + ["segmentations"] + path.split("/")[-2:]).replace("jpg", "png")
                for path in self.all_image_paths
            ]
        )
        self.all_image_classes = np.array([int(path.split("/")[-2].split(".")[0]) for path in self.all_image_paths])
        self.all_image_indices = np.array(
            [int(path.split("/")[-1].split(".")[0].split("_")[-1]) for path in self.all_image_paths]
        )

        # Sort paths and classes by index
        sort_indices = np.argsort(self.all_image_indices)
        self.all_image_indices = self.all_image_indices[sort_indices]
        self.all_image_paths = self.all_image_paths[sort_indices]
        self.all_segmentation_paths = self.all_segmentation_paths[sort_indices]
        self.all_image_classes = self.all_image_classes[sort_indices]

        with open(f"{root}/train_test_split.txt") as f:
            lines = f.readlines()
            self.should_include_mask = np.array([int(line.split(" ")[1]) == 1 for line in lines])
            if not train:
                self.should_include_mask = ~self.should_include_mask

        self.all_image_indices = self.all_image_indices[self.should_include_mask]
        self.all_image_paths = self.all_image_paths[self.should_include_mask]
        self.all_segmentation_paths = self.all_segmentation_paths[self.should_include_mask]
        self.all_image_classes = self.all_image_classes[self.should_include_mask]

        self.transform = transform
        self.semantic = semantic

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        image_path = self.all_image_paths[idx]
        segmentation_path = self.all_segmentation_paths[idx]

        image = np.array(Image.open(image_path)) / 255
        if len(image.shape) != 3:
            image = np.stack([image] * 3, axis=-1)
        image = image.transpose(2, 0, 1)
        target = np.array(Image.open(segmentation_path).convert("L")) / 255

        if self.semantic:
            target_class = self.all_image_classes[idx]
            expanded = np.zeros((201, *target.shape[0]))
            expanded[0] = 1 - target
            expanded[target_class] = target
            target = expanded
        else:
            target = target[np.newaxis]

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


def get_dataset(
    root="data/CUB_200",
    batch_size=64,
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
    num_workers=4,
    ds_size=-1,
):
    data_dir = "/".join(root.split("/")[:-1])
    downloaded = os.path.exists(root)
    if not downloaded:
        urllib.request.urlretrieve(
            "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
            f"{data_dir}/CUB_200_images.tgz",
            reporthook=DownloadProgressBar(),
        )
        urllib.request.urlretrieve(
            "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1",
            f"{data_dir}/CUB_200_segmentations.tgz",
            reporthook=DownloadProgressBar(),
        )
        with tarfile.open(f"{data_dir}/CUB_200.tgz", "r:gz") as tar:
            tar.extractall(data_dir)
        shutil.move(f"{data_dir}/CUB_200_2011", root)
        with tarfile.open(f"{data_dir}/CUB_200_segmentations.tgz", "r:gz") as tar:
            tar.extractall(data_dir)
        shutil.move(f"{data_dir}/segmentations", f"{root}/segmentations")
        os.remove(f"{data_dir}/CUB_200_images.tgz")
        os.remove(f"{data_dir}/CUB_200_segmentations.tgz")

    train_ds = CUB200SegmentationDataset(f"{data_dir}/CUB_200", train=True, transform=train_transform)
    val_ds = CUB200SegmentationDataset(f"{data_dir}/CUB_200", train=False, transform=val_transform)

    if ds_size > 0:
        train_ds = torch.utils.data.Subset(train_ds, torch.arange(ds_size))
        val_ds = torch.utils.data.Subset(val_ds, torch.arange(ds_size))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def inv_normalize(x):
    x = (
        x.permute(0, 2, 3, 1) * torch.tensor(perceptual_mean).to(x.device) + torch.tensor(perceptual_std).to(x.device)
    ).permute(0, 3, 1, 2)
    return x


def generate_sample_output(imgs, preds, targets):
    imgs = inv_normalize(imgs)
    grid = create_image_grid(imgs, preds, targets)
    grid = grid.permute(1, 2, 0).numpy().clip(0, 1)
    return grid
