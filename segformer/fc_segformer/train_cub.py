from .fc_segformer import FC_SegformerForSemanticSegmentation, FC_SegformerConfig
import os
from torchmetrics import MeanAbsoluteError
from transformers import SegformerForSemanticSegmentation, SegformerConfig

import cub_utils
from segmentation_utils import SegmentationTrainer, SegmentationTrainerConfig
from cub_utils import get_dataset
import wandb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import json

if __name__ == "__main__":

    class FCT_Segmentor(SegmentationTrainer):
        def __init__(self, **kwargs):
            self.config = FC_SegformerConfig(**kwargs["architecture"])
            self.model = FC_SegformerForSemanticSegmentation(self.config)
            kwargs["model"] = self.model
            super().__init__(**kwargs)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, test_loader = get_dataset(batch_size=192)

    config_dict = {
        "num_labels": 1,
        "num_encoder_blocks": 4,
        "depths": [2, 2, 2, 2],
        "sr_ratios": [8, 4, 2, 1],
        "hidden_sizes": [32, 64, 160, 256],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "num_attention_heads": [1, 2, 5, 8],
        "mlp_ratios": [4, 4, 4, 4],
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "drop_path_rate": 0.1,
        "layer_norm_eps": 1e-6,
        "decoder_hidden_size": 256,
        "semantic_loss_ignore_index": 255,
        "input_resolution": (256, 256),
        "attention_kernel_size": 3,
        "feedforward_kernel_size": 3,
        "decoder_kernel_size": 3,
    }

    model_config = FC_SegformerConfig(**config_dict)

    effective_batch_size = 192
    true_batch_size = 96
    iters_to_accumulate = effective_batch_size // true_batch_size
    train_loader, val_loader, test_loader = get_dataset(batch_size=true_batch_size)

    model = FC_SegformerForSemanticSegmentation(model_config).to(device)

    mae = MeanAbsoluteError().to(device)
    inv_mae = lambda y_hat, y: 1 - mae(y_hat, y)

    trainer_config = SegmentationTrainerConfig(
        ignore_index=None,
        accuracy_fn=inv_mae,
        lr=0.00006,
        loss_fn=torch.nn.MSELoss(),
        device=device,
        sample_output_fn=cub_utils.generate_sample_output,
        model_config=model_config,
        iters_to_accumulate=iters_to_accumulate,
    )
    trainer = SegmentationTrainer(model=model, config=trainer_config)

    should_resume = False
    run_id = "n5cpaoug" if should_resume else None
    wandb.init(project="segformer", config=config_dict, id=run_id, resume="must" if should_resume else "never")

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

    trainer.fit(train_loader, val_loader, test_loader, n_epochs=5000, test_freq=10, start_epoch=start_epoch)
