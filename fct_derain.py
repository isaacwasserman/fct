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


def blah(x):
    print(x.mean(), x.std())
    return x


# Define the dataset
# class Rain13K(torch.utils.data.Dataset):
#     def __init__(self, root, split, transform=None):
#         # Get absolute path of current python file
#         this_file = os.path.abspath(__file__)
#         this_dir = os.path.dirname(this_file)
#         root = os.path.join(this_dir, root)
#         if split == "train":
#             self.base_dir = os.path.join(root, split, "Rain13K")
#         else:
#             self.base_dir = os.path.join(root, split, "Test100")
#         self.input_dir = os.path.join(self.base_dir, "input")
#         self.target_dir = os.path.join(self.base_dir, "target")
#         self.transform = transform
#         self.length = len(os.listdir(self.input_dir))

#         if split == "train":
#             self.image_ids = glob.glob(os.path.join(self.input_dir, "*.jpg"))
#         elif split == "test":
#             self.image_ids = glob.glob(os.path.join(self.input_dir, "*.png"))
#         self.image_ids = [image_id.split("/")[-1].split(".")[0] for image_id in self.image_ids]

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         x_path = list(glob.glob(os.path.join(self.input_dir, f"{self.image_ids[idx]}.*")))[0]
#         y_path = list(glob.glob(os.path.join(self.target_dir, f"{self.image_ids[idx]}.*")))[0]
#         x = torchvision.io.read_image(x_path).float()
#         y = torchvision.io.read_image(y_path).float()
#         x = self.transform(x)
#         y = self.transform(y)
#         return x, y


# def get_dataset(batch_size=64, train_transform=None, test_transform=None, num_workers=4):
#     # Check if directory data/rain13k/train and data/rain13k/test exists
#     if not os.path.exists("data/Rain13K/train") or not os.path.exists("data/Rain13K/test"):
#         # Make directory data/rain13k
#         os.makedirs("data/Rain13K", exist_ok=True)
#         # gdown.download("https://drive.google.com/uc?id=14BidJeG4nSNuFNFDf99K-7eErCq4i47t", "data/Rain13K/train.zip")
#         # gdown.download("https://drive.google.com/uc?id=1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8", "data/Rain13K/test.zip")
#         os.system("wget https://storage.googleapis.com/thesis_cloud_files/Rain13K/train.zip")
#         os.system("wget https://storage.googleapis.com/thesis_cloud_files/Rain13K/test.zip")
#         os.system("mv train.zip data/Rain13K/")
#         os.system("mv test.zip data/Rain13K/")
#         # Extract contents of train.zip and test.zip to data/rain13k
#         with zipfile.ZipFile("data/Rain13K/train.zip", "r") as zip_ref:
#             zip_ref.extractall("data/Rain13K")
#         with zipfile.ZipFile("data/Rain13K/test.zip", "r") as zip_ref:
#             zip_ref.extractall("data/Rain13K")

#     # Download the dataset
#     train_dataset = Rain13K(root="data/Rain13K", split="train", transform=train_transform)
#     train_dataset, val_dataset = torch.utils.data.random_split(
#         train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
#     )
#     test_dataset = Rain13K(root="data/Rain13K", split="test", transform=test_transform)

#     # Define the dataloaders
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
#     )
#     return train_loader, val_loader, test_loader


def get_dataset(
    ds_root="data/Rain13K.hdf5", batch_size=64, train_transform=None, test_transform=None, num_workers=4
):
    ds = h5py.File(ds_root, "r")
    train_ds = ds["train"]
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    test_ds = ds["test"]

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
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

    def log_test(self, epoch, step=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                batch = next(iter(self.test_loader))
                imgs = batch[:, 0, :, :, :]
                targets = batch[:, 1, :, :, :]
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                preds = self.model(imgs)
                # Inverse normalization
                imgs = self.inverse_normalization(imgs)
                preds = self.inverse_normalization(preds)
                targets = self.inverse_normalization(targets)
                grid = create_image_grid_denoise(imgs, preds, targets, grid_size=(8, 2))
                grid = torch.tensor(grid).permute(2, 0, 1)
                t = epoch * len(self.train_loader) + step
                self.writer.add_image("Test", grid, t)

    def validate(self, epoch, debug_steps=-1):
        accumulated_loss = 0
        accumulated_accuracy = 0
        steps_per_epoch = len(self.val_loader)
        for batch_idx, batch in tqdm(enumerate(self.val_loader), desc=f"Validation {epoch+1}", total=steps_per_epoch):
            if batch_idx > debug_steps > 0:
                break
            if device in ["cuda", "xpu", "privateuseone"]:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    imgs = batch[:, 0, :, :, :]
                    labels = batch[:, 1, :, :, :]
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    preds = self.model(imgs)
                    accumulated_loss += self.calculate_loss(preds, labels)
                    accumulated_accuracy += self.calculate_psnr(preds, labels)
            else:
                imgs = batch[:, 0, :, :, :]
                labels = batch[:, 1, :, :, :]
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = self.model(imgs)
                accumulated_loss += self.calculate_loss(preds, labels)
                accumulated_accuracy += self.calculate_psnr(preds, labels)
        accumulated_loss /= len(self.val_loader)
        accumulated_accuracy /= len(self.val_loader)

        self.writer.add_scalar("Loss/val", accumulated_loss, epoch)
        self.writer.add_scalar("Accuracy/val", accumulated_accuracy, epoch)

        self.checkpoint(accumulated_accuracy)

    def train_epoch(self, optimizer, scaler, epoch, test_freq=0.1, log_freq=0.01, debug_steps=-1):
        steps_per_epoch = len(self.train_loader)
        test_freq = max(1, int(steps_per_epoch * test_freq)) if test_freq > 0 else -1
        log_freq = max(1, int(steps_per_epoch * log_freq)) if log_freq > 0 else -1
        # accumulated_loss = 0
        # accumulated_accuracy = 0
        for batch_idx, batch in tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch+1}", total=steps_per_epoch):
            if batch_idx > debug_steps > 0:
                break
            if device in ["cuda", "xpu", "privateuseone"]:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    batch = batch.to(device, non_blocking=True)
                    imgs = batch[:, 0, :, :, :]
                    labels = batch[:, 1, :, :, :]
                    preds = self.model(imgs)
                    loss = self.calculate_loss(preds, labels)
                    with torch.no_grad():
                        accuracy = self.calculate_psnr(preds, labels)
                        if batch_idx % test_freq == 0 and test_freq > 0:
                            self.log_test(epoch, step=batch_idx)
            else:
                batch = batch.to(device, non_blocking=True)
                imgs = batch[:, 0, :, :, :]
                labels = batch[:, 1, :, :, :]
                preds = self.model(imgs)
                loss = self.calculate_loss(preds, labels)
                with torch.no_grad():
                    accuracy = self.calculate_psnr(preds, labels)
                    if batch_idx % test_freq == 0 and test_freq > 0:
                        self.log_test(epoch, step=batch_idx)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


            step = epoch * steps_per_epoch + batch_idx
            self.writer.add_scalar("Loss/train", loss, step)
            self.writer.add_scalar("Accuracy/train", accuracy, step)

            # accumulated_loss += loss
            # accumulated_accuracy += accuracy

            # if batch_idx % log_freq == 0 and log_freq > 0:
            #     accumulated_loss /= log_freq
            #     accumulated_accuracy /= log_freq
            #     self.writer.add_scalar("Loss/train", accumulated_loss, step)
            #     self.writer.add_scalar("Accuracy/train", accumulated_accuracy, step)
            #     accumulated_loss = 0
            #     accumulated_accuracy = 0

    def fit(self, train_loader, val_loader, test_loader, n_epochs=1, test_freq=0.1, log_freq=0.1, debug_steps=-1):
        fused = device in ["cuda", "xpu", "privateuseone"]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, fused=fused)
        scaler = torch.cuda.amp.GradScaler()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        for epoch in range(self.start_epoch, n_epochs):
            self.train_epoch(optimizer, scaler, epoch, test_freq=test_freq, log_freq=log_freq, debug_steps=debug_steps)
            with torch.no_grad():
                self.validate(epoch, debug_steps=debug_steps)
                self.log_test(epoch)


def divide_by_255(x):
    return x / 255


if __name__ == "__main__":

    from line_profiler import LineProfiler

    # Ensure that all GPU memory is released
    kill_defunct_processes()

    L.seed_everything(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = False

    torch.set_float32_matmul_precision("medium")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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

    light_vit_kwargs = {
        "embed_dim": 32,
        "hidden_dim": 64,
        "q_dim": 64,
        "v_dim": 32,
        "num_heads": 4,
        "num_layers": 4,
        "num_channels": 3,
        "num_classes": 3,
        "dropout": 0.2,
        "patch_equivalent_mode": False,
        "input_resolution": (256, 256),
        "transformer_kernel_size": 3,
        "inverse_normalization": inv_normalize,
        "resume_from_run": "ViT_005",
        "start_epoch": 2,
    }

    def go():
        model_kwargs = light_vit_kwargs

        train_loader, val_loader, test_loader = get_dataset(
            batch_size=16, train_transform=transform, test_transform=transform, num_workers=num_workers
        )
        vit = ViT(**model_kwargs)

        vit.fit(train_loader, val_loader, test_loader, n_epochs=180, test_freq=0.05, log_freq=0.2)

    go()
