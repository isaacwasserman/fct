import os
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
from PIL import Image
from tqdm.auto import tqdm
import warnings
import wandb
from torchmetrics.classification import MulticlassAccuracy
import transformers

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

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


def count_classes_vectorized(labels, num_classes):
    batch_size = labels.shape[0]

    # Flatten the labels tensor and create a batch index tensor
    flat_labels = labels.reshape(batch_size, -1)
    batch_index = torch.arange(batch_size, device=labels.device).unsqueeze(1).expand_as(flat_labels)

    # Create a tensor of shape (batch_size * height * width, 2)
    # where each row is [batch_index, label]
    index_label = torch.stack([batch_index.reshape(-1), flat_labels.reshape(-1)], dim=1)

    # Use sparse tensor to efficiently count occurrences
    counts = torch.sparse_coo_tensor(
        index_label.t(),
        torch.ones(index_label.shape[0], dtype=torch.int64, device=labels.device),
        size=(batch_size, num_classes),
    ).to_dense()

    return counts


ds_mean = [0.49139968, 0.48215841, 0.44653091]
ds_std = [0.24703223, 0.24348513, 0.26158784]


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
    image = transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(image)
    # Normalize
    image = torchvision.transforms.functional.normalize(image, ds_mean, ds_std)
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
    train_ds = torchvision.datasets.VOCSegmentation(
        "data", year="2012", image_set="train", download=not downloaded, transforms=train_transform
    )
    val_ds = torchvision.datasets.VOCSegmentation(
        "data", year="2012", image_set="val", download=not downloaded, transforms=val_transform
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
    x = (x.permute(0, 2, 3, 1) * torch.tensor(ds_mean).to(device) + torch.tensor(ds_std).to(device)).permute(0, 3, 1, 2)
    return x


def differentiable_bincount(input_tensor, minlength):
    index_tensor = input_tensor
    source_tensor = torch.ones_like(input_tensor, device=input_tensor.device)
    result = torch.zeros(minlength, dtype=torch.int64, device=input_tensor.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = torch.scatter_reduce(result, 0, index_tensor, source_tensor, reduce="sum")
    return result


def generate_balanced_cross_entropy(train_loader):
    class_counts = torch.zeros(21)
    for X, Y in train_loader:
        class_counts += torch.bincount(Y.flatten(), minlength=256)[:21]
    class_weights = 1 / class_counts
    class_weights[torch.isinf(class_weights)] = 0
    class_weights = torch.nn.functional.normalize(class_weights, p=1, dim=0).to(device)
    class_weights[torch.isinf(class_weights)] = 0
    class_weights[torch.isnan(class_weights)] = 0

    def balanced_cross_entropy(y_hat, y):
        return torch.nn.functional.cross_entropy(y_hat, y, weight=class_weights, ignore_index=255)

    return balanced_cross_entropy


def generate_pascal_sample_output(imgs, preds, targets):
    imgs = inv_normalize(imgs)
    grid = create_image_grid_pascal(imgs, preds, targets)
    grid = grid.permute(1, 2, 0).numpy().clip(0, 1)
    return grid


class PascalTrainer:
    def __init__(self, **hyperparams):
        self.model = hyperparams.get("model", None).to(device)
        # Initialize member variables
        self.inverse_normalization = hyperparams.get("inverse_normalization", inv_normalize)
        self.loss_fn = hyperparams.get("loss_fn", torch.nn.CrossEntropyLoss(ignore_index=255))
        self.accuracy_fn = hyperparams.get(
            "accuracy_fn", MulticlassAccuracy(21, average="weighted", ignore_index=255).to(device)
        )
        self.lr = hyperparams.get("lr", 3e-4)
        self.sample_output_fn = hyperparams.get("sample_output_fn", generate_pascal_sample_output)
        self.current_epoch = 0
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, fused=False  # device in ["cuda", "xpu", "privateuseone"]
        )
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def calculate_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def calculate_accuracy(self, y_hat, y):
        return self.accuracy_fn(y_hat, y)

    def checkpoint(self, accuracy):
        """Saves the model to checkpoints directory."""
        checkpoint_dir = f"checkpoints/{wandb.run.id}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "optimizer": self.optimizer, "epoch": self.epoch},
            f"{checkpoint_dir}/vit.pth",
        )

    def forward(self, imgs):
        preds = self.model(imgs)
        if isinstance(preds, torch.Tensor):
            pass
        elif isinstance(preds, transformers.utils.ModelOutput):
            preds = preds.logits
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = torch.nn.functional.interpolate(preds, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
        return preds

    def log_test(self):
        """Generates a batch of predictions and logs them to wandb."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                imgs, targets = next(iter(self.test_loader))
                imgs = imgs.to(device, non_blocking=True)
                preds = self.forward(imgs).cpu()
                sample_test_outputs = self.sample_output_fn(imgs, preds, targets)

                imgs, targets = next(iter(self.train_loader))
                imgs = imgs.to(device, non_blocking=True)
                preds = self.forward(imgs).cpu()
                sample_train_outputs = self.sample_output_fn(imgs, preds, targets)
                wandb.log(
                    {
                        "Sample Train Outputs": wandb.Image(sample_train_outputs),
                        "Sample Test Outputs": wandb.Image(sample_test_outputs),
                    }
                )

    def validate(self):
        accumulated_loss = 0
        accumulated_accuracy = 0
        for batch_idx, batch in tqdm(enumerate(self.val_loader), desc=f"Validation {self.epoch+1}", total=len(self.val_loader)):
            with torch.autocast(device_type=device, dtype=torch.float16):
                imgs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                preds = self.forward(imgs)
                accumulated_loss += self.calculate_loss(preds, labels)
                accumulated_accuracy += self.calculate_accuracy(preds, labels)
        accumulated_loss /= len(self.val_loader)
        accumulated_accuracy /= len(self.val_loader)
        wandb.log({"val_accuracy": accumulated_accuracy, "val_loss": accumulated_loss})
        return accumulated_accuracy

    def train_epoch(self):
        for batch_idx, batch in tqdm(enumerate(self.train_loader), desc=f"Epoch {self.epoch+1}", total=len(self.train_loader)):
            with torch.autocast(device_type=device, dtype=torch.float16):
                imgs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                preds = self.forward(imgs)
                loss = self.calculate_loss(preds, labels)
                with torch.no_grad():
                    accuracy = self.calculate_accuracy(preds, labels)
                    if batch_idx % self.test_freq == 0 and self.test_freq > 0:
                        self.log_test()

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            wandb.log({"train_accuracy": accuracy, "train_loss": loss})

    def fit(self, train_loader, val_loader, test_loader, n_epochs=1, test_freq=0.1, start_epoch=0):
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.test_freq = test_freq
        for self.epoch in range(start_epoch, n_epochs):
            self.train_epoch()
            with torch.no_grad():
                val_accuracy = self.validate()
                self.checkpoint(val_accuracy)
