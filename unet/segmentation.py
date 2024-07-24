import torch
from torchmetrics.classification import MulticlassAccuracy
import wandb
import pascal_utils
from .unet import UNetForSemanticSegmentation, UNetConfig


class UNet_Segmentor(pascal_utils.PascalTrainer):
    def __init__(self, **kwargs):
        self.config = UNetConfig(**kwargs["architecture"])
        self.model = UNetForSemanticSegmentation(self.config)
        kwargs["model"] = self.model
        super().__init__(**kwargs)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def go():
        train_loader, val_loader, test_loader = pascal_utils.get_dataset(batch_size=4)

        model_kwargs = {
            "architecture": {
                "num_labels": 21,
                "input_resolution": (256, 256),
            },
            "loss_fn": pascal_utils.generate_balanced_cross_entropy(train_loader),
            "accuracy_fn": MulticlassAccuracy(21, average="weighted", ignore_index=255).to(device),
            "lr": 0.00006,
        }

        segmentor = UNet_Segmentor(**model_kwargs)

        should_resume = False
        run_id = "5oydzpol" if should_resume else None
        wandb.init(project="fct_unet", config=model_kwargs, id=run_id, resume="must" if should_resume else "never")

        start_epoch = 0
        if should_resume:
            checkpoint_dir = f"checkpoints/{run_id}"
            checkpoint = torch.load(checkpoint_dir + "/vit.pth")
            start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            segmentor.optimizer = checkpoint["optimizer"]
            segmentor.optimizer.add_param_group({"params": segmentor.model.parameters()})
            segmentor.model.load_state_dict(state_dict)

        segmentor.fit(train_loader, val_loader, test_loader, n_epochs=5000, test_freq=10, start_epoch=start_epoch)

    go()
