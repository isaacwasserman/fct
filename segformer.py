import os
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)

from pascal_utils import PascalTrainer, get_dataset
import wandb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    configuration_dict = {
        "num_labels": 21,
    }

    configuration = SegformerConfig(**configuration_dict)

    train_loader, val_loader, test_loader = get_dataset(batch_size=64)

    model = SegformerForSemanticSegmentation(configuration)

    trainer = PascalTrainer(model=model, lr=0.00006)

    wandb.init(project="segformer", config=configuration_dict)

    trainer.fit(train_loader, val_loader, test_loader, n_epochs=5000, test_freq=10)
