import os
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from fct.segmentation_utils import SegmentationTrainer, get_dataset
import wandb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import json

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    # torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    configuration_dict = {
        "num_labels": 21,
    }

    configuration = SegformerConfig(**configuration_dict)

    train_loader, val_loader, test_loader = get_dataset(batch_size=64)

    model = SegformerForSemanticSegmentation(configuration)

    trainer = SegmentationTrainer(model=model, lr=0.00006)

    should_resume = True
    run_id = "2vhhtkyg" if should_resume else None
    wandb.init(project="segformer", config=configuration_dict, id=run_id, resume="must" if should_resume else "never")

    start_epoch = 0
    if should_resume:
        checkpoint_dir = f"checkpoints/{run_id}"
        checkpoint = torch.load(checkpoint_dir + "/vit.pth")
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            trainer.optimizer = checkpoint["optimizer"]
            trainer.optimizer.add_param_group({"params": trainer.model.parameters()})
        else:
            state_dict = checkpoint
            with open(checkpoint_dir + "/metadata.json", "r") as f:
                metadata = json.load(f)
                start_epoch = metadata["epoch"]
        trainer.model.load_state_dict(state_dict)

    trainer.fit(train_loader, val_loader, test_loader, n_epochs=5000, test_freq=10)
