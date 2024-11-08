import torch
import torch.utils
import torchvision
import torchvision.transforms.v2 as transforms_v2
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import os
import numpy as np
from einops import rearrange
from utils import *
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from fct import *
import warnings
from torchmetrics.classification import MulticlassAccuracy
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
import wandb
import segmentation_models_pytorch as smp
import json


def get_dataset(batch_size=64, train_transform=None, val_transform=None, test_transform=None, num_workers=4, ds_size=-1):
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




class ViT_Segmentor:
    def __init__(self, **hyperparams):
        super().__init__()
        # Instantiate model
        backbone = torchvision.models.vit_b_16(weights=None, image_size=hyperparams["input_resolution"][0], num_classes=hyperparams["num_classes"])
        # Override .forward() method
        def backbone_forward(self, x):
            # Reshape and permute the input tensor
            x = self._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.encoder(x)

            # print(x.shape)
            x = x[:, 1:]
            x = x.reshape(n, 16, 16, -1)
            x = x.permute(0, 3, 1, 2)

            # # Classifier "token" as used by standard language architectures
            # x = x[:, 0]

            # x = self.heads(x)
            x = torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=True)
            return x
        backbone.forward = backbone_forward.__get__(backbone)
        output_module = torch.nn.Sequential(
            torch.nn.Conv2d(
                768, hyperparams["num_classes"], kernel_size=3, padding=1
            ),
        )
        self.model = torch.nn.Sequential(backbone, output_module).to(device)

        # Initialize member variables
        self.inverse_normalization = hyperparams.get("inverse_normalization", lambda x: x)
        self.loss_fn = hyperparams.get("loss_fn", torch.nn.CrossEntropyLoss())
        self.accuracy_fn = hyperparams.get("accuracy_fn", torch.nn.CrossEntropyLoss())
        self.lr = hyperparams.get("lr", 3e-4)
        self.sample_output_fn = hyperparams.get("sample_output_fn", lambda x: None)
        self.current_epoch = 0
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, fused=device in ["cuda", "xpu", "privateuseone"]
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

    def log_test(self):
        """Generates a batch of predictions and logs them to wandb."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                imgs, targets = next(iter(self.test_loader))
                imgs = imgs.to(device, non_blocking=True)
                preds = self.model(imgs).cpu()
                sample_test_outputs = self.sample_output_fn(imgs, preds, targets)

                imgs, targets = next(iter(self.train_loader))
                imgs = imgs.to(device, non_blocking=True)
                preds = self.model(imgs).cpu()
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
                preds = self.model(imgs)
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
                preds = self.model(imgs)
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


if __name__ == "__main__":
    kill_defunct_processes()
    torch.set_float32_matmul_precision("medium")
    # torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    print("Note: bias is disabled because it's exploding")

    def go():
        train_loader, val_loader, test_loader = get_dataset(
            batch_size=4,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            num_workers=4,
            ds_size=-1,
        )

        model_kwargs = {
            "embed_dim": 64,
            "hidden_dim": 128,
            "q_dim": 128,
            "v_dim": 64,
            "num_heads": 4,
            "num_layers": 8,
            "num_channels": 3,
            "num_classes": 21,
            "dropout": 0.2,
            "patch_equivalent_mode": False,
            "input_resolution": (256, 256),
            "transformer_kernel_size": [5, 3, 1],
            "inverse_normalization": inv_normalize,
            "loss_fn": generate_balanced_cross_entropy(train_loader),
            "accuracy_fn": MulticlassAccuracy(21, average="weighted", ignore_index=255).to(device),
            "lr": 0.00005,
            "sample_output_fn": generate_pascal_sample_output,
        }

        vit = ViT_Segmentor(**model_kwargs)

        should_resume = False
        run_id = "4194toix" if should_resume else None
        wandb.init(project="vit_segmentation", config=model_kwargs, id=run_id, resume="must" if should_resume else "never")

        start_epoch = 0
        if should_resume:
            checkpoint_dir = f"checkpoints/{run_id}"
            checkpoint = torch.load(checkpoint_dir + "/vit.pth")
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
                state_dict = checkpoint["state_dict"]
                vit.optimizer = checkpoint["optimizer"]
                vit.optimizer.add_param_group({"params": vit.model.parameters()})
            else:
                state_dict = checkpoint
                with open(checkpoint_dir + "/metadata.json", "r") as f:
                    metadata = json.load(f)
                    start_epoch = metadata["epoch"]
            vit.model.load_state_dict(state_dict)

        vit.fit(train_loader, val_loader, test_loader, n_epochs=5000, test_freq=10, start_epoch=start_epoch)

    go()
