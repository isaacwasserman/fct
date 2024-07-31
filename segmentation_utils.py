import os
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transforms_v2
from PIL import Image
from tqdm.auto import tqdm
import warnings
import wandb
from torchmetrics.classification import MulticlassAccuracy
import transformers

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)


def count_classes_vectorized(labels, num_classes):
    batch_size = labels.shape[0]

    # Flatten the labels tensor and create a batch index tensor
    flat_labels = labels.reshape(batch_size, -1)
    batch_index = torch.arange(batch_size, device=labels.device).unsqueeze(1).expand_as(flat_labels)

    # Create a tensor of shape (batch_size * height * width, 2)
    # where each row is [batch_index, label]
    index_label = torch.stack([batch_index.reshape(-1), flat_labels.reshape(-1)], dim=1)

    # Use sparse tensor to efficiently count occurrences
    counts = torch.sparse_coo_tensor(
        index_label.t(),
        torch.ones(index_label.shape[0], dtype=torch.int64, device=labels.device),
        size=(batch_size, num_classes),
    ).to_dense()

    return counts


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


class SegmentationTrainerConfig:
    def __init__(
        self,
        ignore_index=None,
        accuracy_fn=None,
        loss_fn=None,
        device=device,
        lr=3e-4,
        sample_output_fn=None,
        model_config=None,
    ):
        self.device = device
        self.ignore_index = ignore_index
        self.accuracy_fn = (
            accuracy_fn
            if accuracy_fn
            else MulticlassAccuracy(model_config.num_labels, average="weighted", ignore_index=ignore_index).to(device)
        )
        self.loss_fn = loss_fn if loss_fn else torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lr = lr
        self.sample_output_fn = sample_output_fn
        if self.sample_output_fn is None:
            raise ValueError("sample_output_fn must be provided")
        self.model_config = model_config

# @torch.compile()
class SegmentationTrainer(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, config: SegmentationTrainerConfig):
        super().__init__()
        self.config = config
        self.model = model
        self.current_epoch = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, fused=False)
        self.grad_scaler = torch.amp.GradScaler(config.device)

    def calculate_loss(self, y_hat, y):
        return self.config.loss_fn(y_hat, y)

    def calculate_accuracy(self, y_hat, y):
        return self.config.accuracy_fn(y_hat, y)

    def checkpoint(self, accuracy):
        """Saves the model to checkpoints directory."""
        checkpoint_dir = f"checkpoints/{wandb.run.id}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "optimizer": self.optimizer, "epoch": self.epoch},
            f"{checkpoint_dir}/vit.pth",
        )

    def forward(self, imgs):
        imgs = imgs.to(self.config.device, non_blocking=True, dtype=torch.float32)
        preds = self.model(imgs)
        if isinstance(preds, torch.Tensor):
            pass
        elif isinstance(preds, transformers.utils.ModelOutput):
            preds = preds.logits
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = torch.nn.functional.interpolate(preds, size=imgs.shape[-2:], mode="bilinear", align_corners=False)
        return preds

    def log_test(self):
        """Generates a batch of predictions and logs them to wandb."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                imgs, targets = next(iter(self.test_loader))
                imgs = imgs.to(device, non_blocking=True)
                preds = self.forward(imgs).cpu()
                sample_test_outputs = self.config.sample_output_fn(imgs, preds, targets)

                imgs, targets = next(iter(self.train_loader))
                imgs = imgs.to(device, non_blocking=True)
                preds = self.forward(imgs).cpu()
                sample_train_outputs = self.config.sample_output_fn(imgs, preds, targets)
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
                preds = self.forward(imgs)
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
                preds = self.forward(imgs)
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
