import torch
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


DATASET_PATH = "data"
CHECKPOINT_PATH = "checkpoints"

L.seed_everything(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.mps.deterministic = True
torch.backends.mps.benchmark = False

torch.set_float32_matmul_precision("medium")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

num_workers = 4

test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
    ]
)
train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
    ]
)


def get_dataset(batch_size=64, ds_len=45000):
    # Download the dataset
    train_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    # Split the dataset
    L.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [ds_len, len(train_dataset) - ds_len])
    L.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [len(train_dataset) - ds_len, ds_len])

    # Define the dataloaders
    torch.random.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
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
        self.classification_head = torch.nn.Sequential(torch.nn.LayerNorm(embed_dim), torch.nn.Linear(embed_dim, num_classes))
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

        # Global Average Pooling
        pooled = x.reshape(B, D, -1).mean(dim=-1)  # (B, D, H, W) -> (B, D, HW) -> (B, D)

        # Classification
        out = self.classification_head(pooled)
        return out


class ViT:
    def __init__(self, **hyperparams):
        super().__init__()
        self.model = VisionTransformer(**model_kwargs).to(device)
        if not os.path.exists("runs"):
            os.makedirs("runs")
        prev_runs = [int(x.split("_")[-1]) for x in os.listdir("runs") if "ViT_" in x] + [-1]
        self.run_id = f"ViT_{max(prev_runs) + 1:03d}"
        self.log_dir = f"runs/{self.run_id}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.best_accuracy = 0

    def calculate_loss(self, y_hat, y):
        return torch.nn.functional.cross_entropy(y_hat, y)

    def calculate_accuracy(self, y_hat, y):
        return (y_hat.argmax(dim=-1) == y).float().mean()

    def checkpoint(self, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.model.state_dict(), f"{self.log_dir}/vit.pth")

    def log_test(self, test_loader, epoch):
        with torch.no_grad():
            batch = next(iter(test_loader))
            imgs, labels = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = self.model(imgs)
            # Inverse normalization
            imgs = imgs.cpu() * torch.tensor([0.24703223, 0.24348513, 0.26158784]).view(1, 3, 1, 1) + torch.tensor(
                [0.49139968, 0.48215841, 0.44653091]
            ).view(1, 3, 1, 1)
            grid = create_image_grid(imgs, preds, labels, grid_size=(16, 16))
            grid = torch.tensor(grid).permute(2, 0, 1)
            self.writer.add_image("Test", grid, epoch)

    def validate(self, val_loader, epoch):
        accumulated_loss = 0
        accumulated_accuracy = 0
        for batch in tqdm(val_loader, desc="Validation"):
            with torch.autocast(device_type=device, dtype=torch.float16):
                imgs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = self.model(imgs)
                accumulated_loss += self.calculate_loss(preds, labels)
                accumulated_accuracy += self.calculate_accuracy(preds, labels)
        accumulated_loss /= len(val_loader)
        accumulated_accuracy /= len(val_loader)

        self.writer.add_scalar("Loss/val", accumulated_loss, epoch)
        self.writer.add_scalar("Accuracy/val", accumulated_accuracy, epoch)

        self.checkpoint(accumulated_accuracy)

    def train_epoch(self, train_loader, optimizer, scaler, epoch):
        steps_per_epoch = len(train_loader)
        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}", total=steps_per_epoch):
            if device in ["cuda", "xpu", "privateuseone"]:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    imgs, labels = batch
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = self.model(imgs)
                    loss = self.calculate_loss(preds, labels)
                    with torch.no_grad():
                        accuracy = self.calculate_accuracy(preds, labels)
            else:
                imgs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = self.model(imgs)
                loss = self.calculate_loss(preds, labels)
                with torch.no_grad():
                    accuracy = self.calculate_accuracy(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step = epoch * steps_per_epoch + batch_idx
            self.writer.add_scalar("Loss/train", loss, step)
            self.writer.add_scalar("Accuracy/train", accuracy, step)

    def fit(self, train_loader, val_loader, n_epochs=1):
        fused = False
        if device in ["cuda", "xpu", "privateuseone"]:
            fused = True
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, fused=fused)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(n_epochs):
            self.train_epoch(train_loader, optimizer, scaler, epoch)

            with torch.no_grad():
                self.validate(val_loader, epoch)
                self.log_test(test_loader, epoch)


if __name__ == "__main__":
    # Ensure that all GPU memory is released
    kill_defunct_processes()

    standard_vit_equivalent_kwargs = {
        "embed_dim": 256,
        "hidden_dim": 512,
        "q_dim": 512,
        "v_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "num_channels": 3,
        "num_classes": 10,
        "dropout": 0.2,
        "patch_equivalent_mode": True,
        "patch_width": 4,
        "input_resolution": (32, 32),
        "transformer_kernel_size": 1,
    }

    model_kwargs = standard_vit_equivalent_kwargs

    train_loader, val_loader, test_loader = get_dataset(batch_size=128, ds_len=45000)
    vit = ViT(**model_kwargs)
    vit.fit(train_loader, val_loader, n_epochs=180)
