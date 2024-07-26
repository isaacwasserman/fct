import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
import numpy as np
import os

pascal_palette = (
    [
        [0, 0, 0],  # Black
        [230, 25, 75],  # Red
        [60, 180, 75],  # Green
        [255, 225, 25],  # Yellow
        [0, 130, 200],  # Blue
        [245, 130, 48],  # Orange
        [145, 30, 180],  # Purple
        [70, 240, 240],  # Cyan
        [240, 50, 230],  # Magenta
        [210, 245, 60],  # Lime
        [250, 190, 190],  # Pink
        [0, 128, 128],  # Teal
        [230, 190, 255],  # Lavender
        [170, 110, 40],  # Brown
        [255, 250, 200],  # Beige
        [128, 0, 0],  # Maroon
        [170, 255, 195],  # Mint
        [128, 128, 0],  # Olive
        [255, 215, 180],  # Coral
        [0, 0, 128],  # Navy
        [128, 128, 128],  # Gray
    ]
    + ([[255, 255, 255]] * 234)
    + [[255, 255, 255]]
)  # White

pascal_palette = torch.tensor(pascal_palette) / 255.0


def create_image_grid_pascal(x, y_hat, y, width=None):
    y_hat_int = y_hat.argmax(dim=1, keepdim=False)
    y_hat_colorized = pascal_palette[y_hat_int].permute(0, 3, 1, 2)
    y_colorized = pascal_palette[y].permute(0, 3, 1, 2)
    lineups = torch.cat([x.cpu(), y_hat_colorized.cpu(), y_colorized.cpu()], dim=3)
    grid = torchvision.utils.make_grid(lineups, nrow=2)
    return grid


perceptual_mean = [0.49139968, 0.48215841, 0.44653091]
perceptual_std = [0.24703223, 0.24348513, 0.26158784]


def transform(image, target, augment=True):
    longest_side = max(image.size[0], image.size[1])

    cropped_size = np.random.randint(int(0.6 * longest_side), longest_side)
    crop_top = np.random.randint(0, longest_side - cropped_size)
    crop_left = np.random.randint(0, longest_side - cropped_size)
    should_flip = np.random.randint(0, 2) == 1

    image = torch.from_numpy(np.array(image)) / 255
    image = image.permute(2, 0, 1)
    target = torch.from_numpy(np.array(target))
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
    # Make integer map
    target = torch.tensor(np.array(target)).long()
    return image, target


train_transform = lambda image, target: transform(image, target, augment=True)
val_transform = lambda image, target: transform(image, target, augment=False)
test_transform = lambda image, target: transform(image, target, augment=False)


def get_dataset(
    batch_size=64,
    train_transform=train_transform,
    val_transform=val_transform,
    test_transform=test_transform,
    num_workers=4,
    ds_size=-1,
):
    downloaded = os.path.exists("data/VOCdevkit/VOC2012")
    try:
        train_ds = torchvision.datasets.VOCSegmentation(
            "data", year="2012", image_set="train", download=not downloaded, transforms=train_transform
        )
        val_ds = torchvision.datasets.VOCSegmentation(
            "data", year="2012", image_set="val", download=not downloaded, transforms=val_transform
        )
    except RuntimeError:
        train_ds = torchvision.datasets.VOCSegmentation(
            "data", year="2012", image_set="train", download=True, transforms=train_transform
        )
        val_ds = torchvision.datasets.VOCSegmentation(
            "data", year="2012", image_set="val", download=True, transforms=val_transform
        )

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


def generate_pascal_sample_output(imgs, preds, targets):
    imgs = inv_normalize(imgs)
    grid = create_image_grid_pascal(imgs, preds, targets)
    grid = grid.permute(1, 2, 0).numpy().clip(0, 1)
    return grid
