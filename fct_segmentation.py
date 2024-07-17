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
from torch.utils.tensorboard import SummaryWriter
from fct import *
import warnings
from torchmetrics.classification import MulticlassAccuracy
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
import wandb
import segmentation_models_pytorch as smp
import json


def get_dataset(batch_size=64, image_transform=None, label_transform=None, num_workers=4, ds_size=-1):
    downloaded = os.path.exists("data/VOCdevkit/VOC2012")
    train_ds = torchvision.datasets.VOCSegmentation(
        "data",
        year="2012",
        image_set="train",
        download=not downloaded,
        transform=image_transform,
        target_transform=label_transform,
    )
    val_ds = torchvision.datasets.VOCSegmentation(
        "data",
        year="2012",
        image_set="val",
        download=not downloaded,
        transform=image_transform,
        target_transform=label_transform,
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


# Define Vision Transformer
class FullyConvolutionalTransformer(torch.nn.Module):
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
        if not isinstance(transformer_kernel_size, list):
            transformer_kernel_size = [transformer_kernel_size] * num_layers
        elif len(transformer_kernel_size) < num_layers:
            transformer_kernel_size = transformer_kernel_size + [transformer_kernel_size[-1]] * (
                num_layers - len(transformer_kernel_size)
            )
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
                    kernel_size=transformer_kernel_size[block_index],
                )
                for block_index in range(num_layers)
            )
        )
        self.dropout = torch.nn.Dropout(dropout)

        self.positional_bias = torch.nn.Parameter(torch.randn((1, embed_dim, *internal_resolution)))

    def forward(self, x):
        # Apply depthwise separable convolution embedding
        x = self.input_layer_cnn(x)  # (B, D, H, W)
        B, D, H, W = x.shape

        # Add positional embedding
        pos_embedding = self.positional_bias.repeat(B, 1, 1, 1)
        x = x + pos_embedding

        # Apply Transforrmer
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class ViT:
    def __init__(self, **hyperparams):
        super().__init__()
        # Instantiate model
        backbone = FullyConvolutionalTransformer(**hyperparams)
        output_module = torch.nn.Sequential(
            torch.nn.LayerNorm((hyperparams["embed_dim"], *hyperparams["input_resolution"])),
            torch.nn.ZeroPad2d(same_padding(hyperparams["transformer_kernel_size"][-1])),
            torch.nn.Conv2d(
                hyperparams["embed_dim"], hyperparams["num_classes"], kernel_size=hyperparams["transformer_kernel_size"][-1]
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
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    print("Note: bias is disabled because it's exploding")

    def go():
        train_loader, val_loader, test_loader = get_dataset(
            batch_size=8, image_transform=image_transform, label_transform=label_transform, num_workers=4, ds_size=-1
        )

        model_kwargs = {
            "embed_dim": 32,
            "hidden_dim": 64,
            "q_dim": 64,
            "v_dim": 32,
            "num_heads": 2,
            "num_layers": 6,
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

        vit = ViT(**model_kwargs)

        should_resume = True
        run_id = "4194toix" if should_resume else None
        wandb.init(project="fct_segmentation", config=model_kwargs, id=run_id, resume="must" if should_resume else "never")

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
