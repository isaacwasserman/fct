import torch
from torchmetrics.classification import MulticlassAccuracy
import wandb
import segmentation_utils
import pascal_utils
from .unet import UNetForSemanticSegmentation, UNetConfig


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def go():
        train_loader, val_loader, test_loader = pascal_utils.get_dataset(batch_size=4)

        model_config = UNetConfig(num_labels=21, input_resolution=(256, 256))

        trainer_config = segmentation_utils.SegmentationTrainerConfig(
            ignore_index=255,
            accuracy_fn=MulticlassAccuracy(21, average="weighted", ignore_index=255).to(device),
            lr=0.00006,
            loss_fn=segmentation_utils.generate_balanced_cross_entropy(train_loader),
            device=device,
            sample_output_fn=pascal_utils.generate_pascal_sample_output,
            model_config=UNetConfig(num_labels=21, input_resolution=(256, 256)),
        )

        model = UNetForSemanticSegmentation(model_config).to(device)
        trainer = segmentation_utils.SegmentationTrainer(model=model, config=trainer_config)

        should_resume = False
        run_id = "nuw1htvc" if should_resume else None
        wandb.init(project="fct_unet", config=trainer_config, id=run_id, resume="must" if should_resume else "never")

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
