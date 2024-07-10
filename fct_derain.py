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


def blah(x):
    print(x.mean(), x.std())
    return x


# Define the dataset
class Rain13k(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        if split == "train":
            self.base_dir = os.path.join(root, split, "Rain13k")
        else:
            self.base_dir = os.path.join(root, split, "Test100")
        self.input_dir = os.path.join(self.base_dir, "input")
        self.target_dir = os.path.join(self.base_dir, "target")
        self.transform = transform
        self.length = len(os.listdir(self.input_dir))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_path = list(glob.glob(os.path.join(self.input_dir, f"{idx+1}.*")))[0]
        y_path = list(glob.glob(os.path.join(self.target_dir, f"{idx+1}.*")))[0]
        x = torchvision.io.read_image(x_path).float()
        y = torchvision.io.read_image(y_path).float()
        x = self.transform(x)
        y = self.transform(y)
        return x, y


def get_dataset(batch_size=64, ds_len=45000, train_transform=None, test_transform=None, num_workers=4):
    # Check if directory data/rain13k/train and data/rain13k/test exists
    if not os.path.exists("data/Rain13k/train") or not os.path.exists("data/Rain13k/test"):
        # Make directory data/rain13k
        os.makedirs("data/Rain13k", exist_ok=True)
        gdown.download("https://drive.google.com/uc?id=14BidJeG4nSNuFNFDf99K-7eErCq4i47t", "data/Rain13k/train.zip")
        gdown.download("https://drive.google.com/uc?id=1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8", "data/Rain13k/test.zip")
        # Extract contents of train.zip and test.zip to data/rain13k
        with zipfile.ZipFile("data/Rain13k/train.zip", "r") as zip_ref:
            zip_ref.extractall("data/Rain13k")
        with zipfile.ZipFile("data/Rain13k/test.zip", "r") as zip_ref:
            zip_ref.extractall("data/Rain13k")

    # Download the dataset
    train_dataset = Rain13k(root="data/Rain13k", split="train", transform=train_transform)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )
    test_dataset = Rain13k(root="data/Rain13k", split="test", transform=test_transform)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
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
        self.denoising_head = torch.nn.Sequential(
            torch.nn.LayerNorm((embed_dim, *internal_resolution)),
            torch.nn.ZeroPad2d(same_padding(transformer_kernel_size)),
            torch.nn.Conv2d(embed_dim, 3, kernel_size=transformer_kernel_size),
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
        out = self.denoising_head(x)
        return out


class ViT:
    def __init__(self, **hyperparams):
        super().__init__()
        self.model = VisionTransformer(**hyperparams).to(device)
        if not os.path.exists("runs"):
            os.makedirs("runs")
        prev_runs = [int(x.split("_")[-1]) for x in os.listdir("runs") if "ViT_" in x] + [-1]
        self.run_id = f"ViT_{max(prev_runs) + 1:03d}"
        self.log_dir = f"runs/{self.run_id}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.best_accuracy = 0
        self.inverse_normalization = hyperparams.get("inverse_normalization", lambda x: x)

    def calculate_loss(self, y_hat, y):
        return (y_hat - y).pow(2).mean()

    def calculate_psnr(self, y_hat, y):
        y_hat = self.inverse_normalization(y_hat) * 255
        y = self.inverse_normalization(y) * 255
        y_hat_luma = y_hat.permute(0, 2, 3, 1) @ torch.tensor([0.299, 0.587, 0.114]).to(device)
        y_luma = y.permute(0, 2, 3, 1) @ torch.tensor([0.299, 0.587, 0.114]).to(device)
        diff = (y_hat_luma - y_luma).abs()
        diff_squared = diff**2
        mse = diff_squared.mean()
        rmse = mse.sqrt()
        psnr = 20 * torch.log10(255 / rmse)
        return psnr

    def checkpoint(self, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.model.state_dict(), f"{self.log_dir}/vit.pth")

    def log_test(self, test_loader, epoch):
        with torch.no_grad():
            batch = next(iter(test_loader))
            imgs, targets = batch
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = self.model(imgs)
            # Inverse normalization
            imgs = self.inverse_normalization(imgs)
            preds = self.inverse_normalization(preds)
            targets = self.inverse_normalization(targets)
            grid = create_image_grid_denoise(imgs, preds, targets, grid_size=(8, 2))
            grid = torch.tensor(grid).permute(2, 0, 1)
            self.writer.add_image("Test", grid, epoch)

    def validate(self, val_loader, epoch):
        accumulated_loss = 0
        accumulated_accuracy = 0
        steps_per_epoch = len(val_loader)
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Validation {epoch+1}", total=steps_per_epoch):
            if batch_idx > 2:
                break
            if device in ["cuda", "xpu", "privateuseone"]:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    imgs, labels = batch
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = self.model(imgs)
                    accumulated_loss += self.calculate_loss(preds, labels)
                    accumulated_accuracy += self.calculate_psnr(preds, labels)
            else:
                imgs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = self.model(imgs)
                accumulated_loss += self.calculate_loss(preds, labels)
                accumulated_accuracy += self.calculate_psnr(preds, labels)
        accumulated_loss /= len(val_loader)
        accumulated_accuracy /= len(val_loader)

        self.writer.add_scalar("Loss/val", accumulated_loss, epoch)
        self.writer.add_scalar("Accuracy/val", accumulated_accuracy, epoch)

        self.checkpoint(accumulated_accuracy)

    def train_epoch(self, train_loader, optimizer, scaler, epoch):
        steps_per_epoch = len(train_loader)
        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}", total=steps_per_epoch):
            if batch_idx > 2:
                break
            if device in ["cuda", "xpu", "privateuseone"]:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    imgs, labels = batch
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = self.model(imgs)
                    loss = self.calculate_loss(preds, labels)
                    with torch.no_grad():
                        accuracy = self.calculate_psnr(preds, labels)
            else:
                imgs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = self.model(imgs)
                loss = self.calculate_loss(preds, labels)
                with torch.no_grad():
                    accuracy = self.calculate_psnr(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step = epoch * steps_per_epoch + batch_idx
            self.writer.add_scalar("Loss/train", loss, step)
            self.writer.add_scalar("Accuracy/train", accuracy, step)

    def fit(self, train_loader, val_loader, test_loader, n_epochs=1):
        fused = device in ["cuda", "xpu", "privateuseone"]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, fused=fused)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(n_epochs):
            self.train_epoch(train_loader, optimizer, scaler, epoch)
            with torch.no_grad():
                self.validate(val_loader, epoch)
                self.log_test(test_loader, epoch)


def divide_by_255(x):
    return x / 255


if __name__ == "__main__":
    from line_profiler import LineProfiler

    # Ensure that all GPU memory is released
    kill_defunct_processes()

    DATASET_PATH = "data"
    CHECKPOINT_PATH = "checkpoints"

    L.seed_everything(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = False

    torch.set_float32_matmul_precision("medium")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"
    print("Device:", device)

    num_workers = 4

    ds_mean = [0.49139968, 0.48215841, 0.44653091]
    ds_std = [0.24703223, 0.24348513, 0.26158784]
    transform = torchvision.transforms.Compose(
        [
            divide_by_255,
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.Normalize(ds_mean, ds_std),
        ]
    )

    def inv_normalize(x):
        x = (x.permute(0, 2, 3, 1) * torch.tensor(ds_mean).to(device) + torch.tensor(ds_std).to(device)).permute(0, 3, 1, 2)
        return x

    # standard_vit_equivalent_kwargs = {
    #     "embed_dim": 256,
    #     "hidden_dim": 512,
    #     "q_dim": 512,
    #     "v_dim": 256,
    #     "num_heads": 8,
    #     "num_layers": 6,
    #     "num_channels": 3,
    #     "num_classes": 10,
    #     "dropout": 0.2,
    #     "patch_equivalent_mode": True,
    #     "patch_width": 4,
    #     "input_resolution": (32, 32),
    #     "transformer_kernel_size": 1,
    # }

    light_vit_kwargs = {
        "embed_dim": 8,
        "hidden_dim": 16,
        "q_dim": 16,
        "v_dim": 8,
        "num_heads": 2,
        "num_layers": 1,
        "num_channels": 3,
        "num_classes": 3,
        "dropout": 0.2,
        "patch_equivalent_mode": False,
        "input_resolution": (256, 256),
        "transformer_kernel_size": 3,
        "inverse_normalization": inv_normalize,
    }

    def go():
        # model_kwargs = standard_vit_equivalent_kwargs
        model_kwargs = light_vit_kwargs

        train_loader, val_loader, test_loader = get_dataset(
            batch_size=1, ds_len=45000, train_transform=transform, test_transform=transform, num_workers=num_workers
        )
        vit = ViT(**model_kwargs)

        vit.fit(train_loader, val_loader, test_loader, n_epochs=1)

    lp = LineProfiler()
    lp.add_function(ViT.train_epoch)
    lp_wrapper = lp(go)
    lp_wrapper()
    lp.print_stats()
