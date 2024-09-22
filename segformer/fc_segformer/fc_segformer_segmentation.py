import torch
from torchmetrics.classification import MulticlassAccuracy
import wandb
import fct.segmentation_utils as segmentation_utils
from fc_segformer import FC_SegformerForSemanticSegmentation, FC_SegformerConfig


class FCT_Segmentor(segmentation_utils.SegmentationTrainer):
    def __init__(self, **kwargs):
        self.config = FC_SegformerConfig(**kwargs["architecture"])
        self.model = FC_SegformerForSemanticSegmentation(self.config)
        kwargs["model"] = self.model
        super().__init__(**kwargs)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    # torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def go():
        train_loader, val_loader, test_loader = segmentation_utils.get_dataset(
            batch_size=64,
            train_transform=segmentation_utils.train_transform,
            val_transform=segmentation_utils.val_transform,
            test_transform=segmentation_utils.test_transform,
            num_workers=4,
            ds_size=-1,
        )

        model_kwargs = {
            "architecture": {
                "num_labels": 21,
                "input_resolution": (256, 256),
            },
            "inverse_normalization": segmentation_utils.inv_normalize,
            "loss_fn": segmentation_utils.generate_balanced_cross_entropy(train_loader),
            "accuracy_fn": MulticlassAccuracy(21, average="weighted", ignore_index=255).to(device),
            "lr": 0.00006,
            "sample_output_fn": segmentation_utils.generate_pascal_sample_output,
        }

        segmentor = FCT_Segmentor(**model_kwargs)

        should_resume = True
        run_id = "q8rolglu" if should_resume else None
        wandb.init(project="fct_segformer", config=model_kwargs, id=run_id, resume="must" if should_resume else "never")

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
