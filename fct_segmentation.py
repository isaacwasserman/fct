import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import os
import numpy as np
from einops import rearrange
from datasets import load_dataset
from utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from fct import *

DATASET_PATH = "/workspace/data/cityscapes"
CHECKPOINT_PATH = "checkpoints"

L.seed_everything(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.mps.deterministic = True
torch.backends.mps.benchmark = False

torch.set_float32_matmul_precision("medium")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

num_workers = 0

# dataset = load_dataset("EduardoPacheco/FoodSeg103")
dataset = load_dataset("Chris1/cityscapes", cache_dir=DATASET_PATH)


def dataset_transform(examples):
    output_size = (256, 256)
    crop_params = transforms.RandomResizedCrop.get_params(torch.ones(output_size), scale=(0.8, 1.0), ratio=(0.9, 1.1))
    should_flip = torch.randint(0, 2, (1,)).item()
    image_transform = transforms.Compose(
        [
            # lambda x: torchvision.transforms.functional.crop(x, *crop_params),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(should_flip),
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
        ]
    )
    examples["image"] = [image_transform(image) for image in examples["image"]]
    label_transform = transforms.Compose(
        [
            # lambda x: torchvision.transforms.functional.crop(x, *crop_params),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(should_flip),
            lambda x: torch.tensor(np.array(x)).squeeze(0).long()[..., 0],
        ]
    )
    examples["label"] = [label_transform(label) for label in examples["semantic_segmentation"]]
    examples = {"image": examples["image"], "label": examples["label"]}
    return examples


dataset.set_transform(dataset_transform)


def get_dataset(batch_size=64, ds_size=None):
    train_set = dataset["train"]
    val_set = dataset["validation"]
    if ds_size is not None:
        train_set = train_set.select(range(ds_size))
        val_set = val_set.select(range(ds_size))
    L.seed_everything(42)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )
    return train_loader, val_loader


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
        self.segmentation_head = torch.nn.Sequential(
            torch.nn.LayerNorm((embed_dim, *input_resolution)),
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

        # Segmentation
        out = self.segmentation_head(x)
        return out


kill_defunct_processes()


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
        y_hat = y_hat.permute(0, 2, 3, 1).reshape(-1, y_hat.shape[1])
        y = y.flatten()
        return (y_hat.argmax(dim=1) == y).float().mean()

    def calculate_miou(self, y_hat, y):
        y_hat = y_hat.argmax(dim=1)
        intersection = torch.logical_and(y_hat == y, y != 0).sum()
        union = torch.logical_or(y_hat == y, y != 0).sum()
        return intersection / union

    def checkpoint(self, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.model.state_dict(), f"{self.log_dir}/vit.pth")

    def visualize_batch(self, images, labels, predictions):
        B = images.shape[0]
        size = 15
        fig, axs = plt.subplots(B, 3, figsize=(size, size * B / 3))
        for i in range(B):
            axs[i, 0].imshow(images[i].permute(1, 2, 0))
            axs[i, 0].axes.get_xaxis().set_visible(False)
            axs[i, 0].axes.get_yaxis().set_visible(False)
            axs[i, 1].imshow(labels[i].squeeze())
            axs[i, 1].axes.get_xaxis().set_visible(False)
            axs[i, 1].axes.get_yaxis().set_visible(False)
            axs[i, 2].imshow(predictions[i].squeeze())
            axs[i, 2].axes.get_xaxis().set_visible(False)
            axs[i, 2].axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        grid_image = get_fig_as_array(fig)
        return grid_image

    def log_test(self, test_loader, epoch):
        with torch.no_grad():
            batch = next(iter(test_loader))
            imgs, labels = batch["image"], batch["label"]
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = self.model(imgs)
            # Inverse normalization
            imgs = imgs.cpu() * torch.tensor([0.24703223, 0.24348513, 0.26158784]).view(1, 3, 1, 1) + torch.tensor(
                [0.49139968, 0.48215841, 0.44653091]
            ).view(1, 3, 1, 1)
            predictions = preds.argmax(1).cpu()
            labels = labels.cpu()
            grid_image = self.visualize_batch(imgs, labels, predictions)
            grid_image = torch.tensor(grid_image).permute(2, 0, 1)
            self.writer.add_image("Test", grid_image, epoch)

    def validate(self, val_loader, epoch):
        accumulated_loss = 0
        accumulated_accuracy = 0
        accumulated_miou = 0
        for batch in tqdm(val_loader, desc="Validation"):
            with torch.autocast(device_type=device, dtype=torch.float16):
                imgs, labels = batch["image"], batch["label"]
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = self.model(imgs)
                accumulated_loss += self.calculate_loss(preds, labels)
                accumulated_accuracy += self.calculate_accuracy(preds, labels)
                accumulated_miou += self.calculate_miou(preds, labels)
        accumulated_loss /= len(val_loader)
        accumulated_accuracy /= len(val_loader)
        accumulated_miou /= len(val_loader)

        self.writer.add_scalar("Loss/val", accumulated_loss, epoch)
        self.writer.add_scalar("Accuracy/val", accumulated_accuracy, epoch)
        self.writer.add_scalar("mIoU/val", accumulated_miou, epoch)

        self.checkpoint(accumulated_accuracy)

    def train_epoch(self, train_loader, optimizer, scaler, epoch):
        steps_per_epoch = len(train_loader)
        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}", total=steps_per_epoch):
            with torch.autocast(device_type=device, dtype=torch.float16):
                imgs, labels = batch["image"], batch["label"]
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = self.model(imgs)
                loss = self.calculate_loss(preds, labels)
                with torch.no_grad():
                    accuracy = self.calculate_accuracy(preds, labels)
                    miou = self.calculate_miou(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step = epoch * steps_per_epoch + batch_idx
            self.writer.add_scalar("Loss/train", loss, step)
            self.writer.add_scalar("Accuracy/train", accuracy, step)
            self.writer.add_scalar("mIoU/train", miou, step)

    def fit(self, train_loader, val_loader, n_epochs=1):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, fused=True)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(n_epochs):
            self.train_epoch(train_loader, optimizer, scaler, epoch)

            with torch.no_grad():
                self.validate(val_loader, epoch)
                self.log_test(val_loader, epoch)


if __name__ == "__main__":

    model_kwargs = {
        "embed_dim": 64,
        "hidden_dim": 128,
        "q_dim": 128,
        "v_dim": 64,
        "num_heads": 8,
        "num_layers": 3,
        "num_channels": 3,
        "num_classes": 34,
        "dropout": 0.2,
        "patch_equivalent_mode": False,
        "input_resolution": (256, 256),
        "transformer_kernel_size": 3,
    }

    train_loader, val_loader = get_dataset(batch_size=8, ds_size=None)
    vit = ViT(**model_kwargs)
    vit.fit(train_loader, val_loader, n_epochs=180)
