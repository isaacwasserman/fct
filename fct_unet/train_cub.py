import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torchmetrics import MeanAbsoluteError
import json
import wandb
import cub_utils
from segmentation_utils import SegmentationTrainer, SegmentationTrainerConfig
from .fct_unet import UNetForSemanticSegmentation, UNetConfig


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, test_loader = cub_utils.get_dataset(batch_size=192)

    config_dict = {
        "max_channels": 128,
        "num_labels": 1,
        "semantic_loss_ignore_index": 255,
        "input_resolution": (256, 256),
        "query_projection_kernel_size": 1,
        "key_projection_kernel_size": 1,
        "value_projection_kernel_size": 1,
        "kv_kernel_size": 3,
        "head_unification_kernel_size": 3,
        "query_projection_stride": 1,
        "key_projection_stride": 1,
        "value_projection_stride": 1,
        "kv_stride": 1,
        "head_unification_stride": 1,
        "query_projection_padding": 0,
        "key_projection_padding": 0,
        "value_projection_padding": 0,
        "kv_padding": 1,
        "head_unification_padding": 1,
        "query_projection_dilation_factor": 0,
        "key_projection_dilation_factor": 0,
        "value_projection_dilation_factor": 0,
        "kv_dilation_factor": 0,
        "head_unification_dilation_factor": 0,
        "use_attention_bias": True,
    }

    model_config = UNetConfig(**config_dict)

    model = UNetForSemanticSegmentation(model_config).to(device)

    effective_batch_size = 192
    true_batch_size = 4
    iters_to_accumulate = effective_batch_size // true_batch_size
    train_loader, val_loader, test_loader = cub_utils.get_dataset(batch_size=true_batch_size)

    mae = MeanAbsoluteError().to(device)
    inv_mae = lambda y_hat, y: 1 - mae(y_hat, y)
    trainer = SegmentationTrainer(
        model=model,
        config=SegmentationTrainerConfig(
            ignore_index=None,
            accuracy_fn=inv_mae,
            lr=0.005,
            loss_fn=torch.nn.MSELoss(),
            device=device,
            sample_output_fn=cub_utils.generate_sample_output,
            model_config=model_config,
            iters_to_accumulate=iters_to_accumulate,
        ),
    )

    should_resume = False
    run_id = "n5cpaoug" if should_resume else None
    wandb.init(project="fct_unet_cub", config=config_dict, id=run_id, resume="must" if should_resume else "never")

    start_epoch = 0
    if should_resume:
        checkpoint_dir = f"checkpoints/{run_id}"
        checkpoint = torch.load(checkpoint_dir + "/vit.pth")
        if "epoch" in checkpoint:
            print("Resuming from epoch", checkpoint["epoch"])
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

    trainer.fit(train_loader, val_loader, test_loader, n_epochs=500, test_freq=10, start_epoch=start_epoch)
