import torch
import torch.utils
import torchvision
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import os
import numpy as np
from einops import rearrange
from utils import *
from tqdm.auto import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter
import time
from fct import *
import gdown
import zipfile
import shutil
import glob
import h5py
import warnings
from line_profiler import LineProfiler
from torchmetrics.classification import MulticlassAccuracy

def get_dataset(batch_size=64, image_transform=None, label_transform=None, num_workers=4):
    downloaded = os.path.exists("data/VOCdevkit/VOC2012")
    train_ds = torchvision.datasets.VOCSegmentation("data", year="2012", image_set="train", download=not downloaded, transform=image_transform, target_transform=label_transform)
    val_ds = torchvision.datasets.VOCSegmentation("data", year="2012", image_set="val", download=not downloaded, transform=image_transform, target_transform=label_transform)

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


# Define Vision Transformer
class VisionTransformer(torch.nn.Module):
    def __init__(
        self,
        embed_dim=256,
        hidden_dim=512,
        q_dim=512,
        v_dim=256,
        num_channels=3,
        num_heads=8,
        num_layers=6,
        num_classes=10,
        dropout=0.0,
        patch_equivalent_mode=True,
        patch_width=4,
        input_resolution=(32, 32),
        transformer_kernel_size=1,
        **kwargs,
    ):
        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            num_classes: Number of classes to predict
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        if patch_equivalent_mode:
            embedding_kernel_size = patch_width
            stride = patch_width
            internal_resolution = (input_resolution[0] // patch_width, input_resolution[1] // patch_width)
        else:
            embedding_kernel_size = 1
            stride = 1
            internal_resolution = input_resolution
        self.input_layer_cnn = torch.nn.Sequential(
            torch.nn.ZeroPad2d(same_padding(embedding_kernel_size) if not patch_equivalent_mode else 0),
            torch.nn.Conv2d(
                num_channels, num_channels, kernel_size=embedding_kernel_size, stride=stride, padding=0, groups=num_channels
            ),
            torch.nn.Conv2d(num_channels, embed_dim, kernel_size=1, stride=1, padding=0),
            torch.nn.GELU(),
        )
        self.transformer = torch.nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    q_dim=q_dim,
                    v_dim=v_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    internal_resolution=internal_resolution,
                    block_index=block_index,
                    kernel_size=transformer_kernel_size,
                )
                for block_index in range(num_layers)
            )
        )
        self.segmentation_head = torch.nn.Sequential(
            torch.nn.LayerNorm((embed_dim, *internal_resolution)),
            torch.nn.ZeroPad2d(same_padding(transformer_kernel_size)),
            torch.nn.Conv2d(embed_dim, num_classes, kernel_size=transformer_kernel_size),
        )
        self.dropout = torch.nn.Dropout(dropout)

        self.positional_bias = torch.nn.Parameter(torch.randn(embed_dim, *internal_resolution))

    def forward(self, x):
        # Apply depthwise separable convolution embedding
        x = self.input_layer_cnn(x)  # (B, D, H, W)
        B, D, H, W = x.shape

        # Add positional embedding
        pos_embedding = self.positional_bias.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, D, H, W)
        x = x + pos_embedding

        # Apply Transforrmer
        x = self.dropout(x)
        x = self.transformer(x)

        # Denoising
        out = self.segmentation_head(x)
        return out


class ViT:
    def __init__(self, **hyperparams):
        super().__init__()
        self.model = VisionTransformer(**hyperparams).to(device)
        if not os.path.exists("runs"):
            os.makedirs("runs")
        prev_runs = [int(x.split("_")[-1]) for x in os.listdir("runs") if "ViT_" in x] + [-1]
        self.run_id = (
            f"ViT_{max(prev_runs) + 1:03d}"
            if hyperparams.get("resume_from_run") is None
            else hyperparams.get("resume_from_run")
        )
        self.log_dir = f"runs/{self.run_id}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.best_accuracy = 0
        self.inverse_normalization = hyperparams.get("inverse_normalization", lambda x: x)
        self.start_epoch = hyperparams.get("start_epoch", 0)
        if hyperparams.get("resume_from_run") is not None:
            self.model.load_state_dict(torch.load(f"{self.log_dir}/vit.pth"))
        self.loss_fn = hyperparams.get("loss_fn", torch.nn.CrossEntropyLoss())
        self.accuracy_fn = hyperparams.get("accuracy_fn", torch.nn.CrossEntropyLoss())
        self.lr = hyperparams.get("lr", 3e-4)

    def calculate_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def calculate_accuracy(self, y_hat, y):
        return self.accuracy_fn(y_hat, y)

    def checkpoint(self, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.model.state_dict(), f"{self.log_dir}/vit.pth")

    def log_test(self, epoch, step=0, save_images=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                imgs, targets = next(iter(self.test_loader))
                imgs = imgs.to(device, non_blocking=True)
                preds = self.model(imgs).cpu()
                # Inverse normalization
                imgs = self.inverse_normalization(imgs)
                grid = create_image_grid_pascal(imgs, preds, targets)
                self.writer.add_image("Test", grid, epoch)
                t = epoch * len(self.train_loader) + step
                self.writer.add_image("Test", grid, t)
                if save_images:
                    plt.imsave(f"{self.log_dir}/test_{t:06d}.png", grid.permute(1, 2, 0).numpy().clip(0, 1))

    def validate(self, epoch, debug_steps=-1):
        accumulated_loss = 0
        accumulated_accuracy = 0
        steps_per_val = len(self.val_loader)
        for batch_idx, batch in tqdm(enumerate(self.val_loader), desc=f"Validation {epoch+1}", total=steps_per_val):
            if batch_idx > debug_steps > 0:
                break
            with torch.autocast(device_type=device, dtype=torch.float16):
                imgs, labels = batch
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                preds = self.model(imgs)
                accumulated_loss += self.calculate_loss(preds, labels)
                accumulated_accuracy += self.calculate_accuracy(preds, labels)
        accumulated_loss /= len(self.val_loader)
        accumulated_accuracy /= len(self.val_loader)

        self.writer.add_scalar("Loss/val", accumulated_loss, epoch)
        self.writer.add_scalar("Accuracy/val", accumulated_accuracy, epoch)

        self.checkpoint(accumulated_accuracy)

    def train_epoch(self, optimizer, scaler, epoch, test_freq=10, log_freq=1, debug_steps=-1):
        steps_per_epoch = len(self.train_loader)
        for batch_idx, batch in tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch+1}", total=steps_per_epoch):
            if batch_idx > debug_steps > 0:
                break
            with torch.autocast(device_type=device, dtype=torch.float16):
                imgs, labels = batch
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                preds = self.model(imgs)
                loss = self.calculate_loss(preds, labels)
                with torch.no_grad():
                    accuracy = self.calculate_accuracy(preds, labels)
                    if batch_idx % test_freq == 0 and test_freq > 0:
                        self.log_test(epoch, step=batch_idx)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            t = epoch * steps_per_epoch + batch_idx
            self.writer.add_scalar("Loss/train", loss, t)
            self.writer.add_scalar("Accuracy/train", accuracy, t)

    def fit(self, train_loader, val_loader, test_loader, n_epochs=1, test_freq=0.1, log_freq=0.1, debug_steps=-1):
        fused = device in ["cuda", "xpu", "privateuseone"]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, fused=fused)
        scaler = torch.cuda.amp.GradScaler()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        for epoch in range(self.start_epoch, n_epochs):
            self.train_epoch(optimizer, scaler, epoch, test_freq=test_freq, log_freq=log_freq, debug_steps=debug_steps)
            with torch.no_grad():
                self.validate(epoch, debug_steps=debug_steps)
                self.log_test(epoch)


if __name__ == "__main__":
    kill_defunct_processes()

    torch.set_float32_matmul_precision("medium")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"
    print("Device:", device)

    num_workers = 4

    ds_mean = [0.49139968, 0.48215841, 0.44653091]
    ds_std = [0.24703223, 0.24348513, 0.26158784]
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.Normalize(ds_mean, ds_std),
        ]
    )
    label_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.Lambda(lambda x: torch.tensor(np.array(x)).long()),
        ]
    )

    def inv_normalize(x):
        x = (x.permute(0, 2, 3, 1) * torch.tensor(ds_mean).to(device) + torch.tensor(ds_std).to(device)).permute(0, 3, 1, 2)
        return x

    def differentiable_bincount(input_tensor, minlength):
        index_tensor = input_tensor
        source_tensor = torch.ones_like(input_tensor, device=input_tensor.device)
        result = torch.zeros(minlength, dtype=torch.int64, device=input_tensor.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = torch.scatter_reduce(result, 0, index_tensor, source_tensor, reduce='sum')
        return result

    torch.autograd.set_detect_anomaly(True)

    def go():
        train_loader, val_loader, test_loader = get_dataset(
            batch_size=16, image_transform=image_transform, label_transform=label_transform, num_workers=num_workers
        )

        class_counts = torch.zeros(21)
        for X, Y in train_loader:
            class_counts += torch.bincount(Y.flatten(), minlength=256)[:21]
        class_weights = (1 / class_counts)
        class_weights = (class_weights / class_weights.sum()).to(device)

        def balanced_cross_entropy(y_hat, y):
            # counts_batch = count_classes_vectorized(y, 256)[:, :21]
            # weights = 1 / counts_batch

            # y[y == 255] = 0
            # counts_batch = torch.vmap(differentiable_bincount, in_dims=0)(y.flatten(1), minlength=21)
        
            # loss_per_image = torch.zeros(y_hat.shape[0], device=y.device)

            # for i in range(y_hat.shape[0]):
            #     loss_per_image[i] = torch.nn.functional.cross_entropy(y_hat[i].unsqueeze(0), y[i].unsqueeze(0), weight=weights[i], ignore_index=255)

            # loss = loss_per_image.mean()
            # return loss

            return torch.nn.functional.cross_entropy(y_hat, y, weight=class_weights, ignore_index=255)

        model_kwargs = {
            "embed_dim": 32,
            "hidden_dim": 64,
            "q_dim": 64,
            "v_dim": 32,
            "num_heads": 4,
            "num_layers": 4,
            "num_channels": 3,
            "num_classes": 21,
            "dropout": 0.2,
            "patch_equivalent_mode": False,
            "input_resolution": (256, 256),
            "transformer_kernel_size": 3,
            "inverse_normalization": inv_normalize,
            "loss_fn": balanced_cross_entropy,
            "accuracy_fn": MulticlassAccuracy(21, average="micro", ignore_index=255).to(device),
            "lr": 3e-4,
            # "resume_from_run": "ViT_005",
            # "start_epoch": 2,
        }

        



        vit = ViT(**model_kwargs)

        vit.fit(train_loader, val_loader, test_loader, n_epochs=180, test_freq=10, log_freq=1)

    go()
