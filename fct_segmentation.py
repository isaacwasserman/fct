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
import pascal_utils

class FCT_Segmentor(pascal_utils.PascalTrainer):
    def __init__(self, **hyperparams):
        backbone = FullyConvolutionalTransformer(**hyperparams)
        output_module = torch.nn.Sequential(
            torch.nn.LayerNorm((hyperparams["embed_dim"], *hyperparams["input_resolution"])),
            torch.nn.ZeroPad2d(same_padding(hyperparams["transformer_kernel_size"][-1])),
            torch.nn.Conv2d(
                hyperparams["embed_dim"], hyperparams["num_classes"], kernel_size=hyperparams["transformer_kernel_size"][-1]
            ),
        )
        self.model = torch.nn.Sequential(backbone, output_module).to(device)
        hyperparams["model"] = self.model
        super().__init__(**hyperparams)


if __name__ == "__main__":
    kill_defunct_processes()
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)
    print("Note: bias is disabled because it's exploding")

    def go():
        train_loader, val_loader, test_loader = pascal_utils.get_dataset(
            batch_size=4,
            train_transform=pascal_utils.train_transform,
            val_transform=pascal_utils.val_transform,
            test_transform=pascal_utils.test_transform,
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
            "inverse_normalization": pascal_utils.inv_normalize,
            "loss_fn": pascal_utils.generate_balanced_cross_entropy(train_loader),
            "accuracy_fn": MulticlassAccuracy(21, average="weighted", ignore_index=255).to(device),
            "lr": 0.00001,
            "sample_output_fn": pascal_utils.generate_pascal_sample_output,
        }

        vit = FCT_Segmentor(**model_kwargs)

        should_resume = False
        run_id = "lp04evzx" if should_resume else None
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
