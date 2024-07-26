import torch
from torchmetrics.regression import MeanAbsoluteError
import wandb
import segmentation_utils
import cub_utils
from .unet import UNetForSemanticSegmentation, UNetConfig


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def go():
        train_loader, val_loader, test_loader = cub_utils.get_dataset(batch_size=4)

        model_config = UNetConfig(num_labels=1, input_resolution=(256, 256), feed_forward_kernel_size=1)

        trainer_config = segmentation_utils.SegmentationTrainerConfig(
            ignore_index=None,
            accuracy_fn=MeanAbsoluteError().to(device),
            lr=0.00006,
            loss_fn=torch.nn.MSELoss(),
            device=device,
            sample_output_fn=cub_utils.generate_sample_output,
            model_config=model_config,
        )

        model = UNetForSemanticSegmentation(model_config).to(device)
        trainer = segmentation_utils.SegmentationTrainer(model=model, config=trainer_config)

        should_resume = False
        run_id = "nuw1htvc" if should_resume else None
        wandb.init(project="fct_unet_simple_cub", config=trainer_config, id=run_id, resume="must" if should_resume else "never")

        start_epoch = 0
        if should_resume:
            checkpoint_dir = f"checkpoints/{run_id}"
            checkpoint = torch.load(checkpoint_dir + "/vit.pth")
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            trainer.optimizer = checkpoint["optimizer"]
            trainer.optimizer.add_param_group({"params": trainer.model.parameters()})
            trainer.model.load_state_dict(state_dict)

        trainer.fit(train_loader, val_loader, test_loader, n_epochs=5000, test_freq=10, start_epoch=start_epoch)

    go()
