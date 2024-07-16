import torch
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import PIL.Image
import cv2
import lightning as L
import torchvision


def print_memory(prefix=""):
    print(f"{prefix} Peak memory used: {torch.cuda.memory_allocated(0)/1024/1024/1024} GiB")


def same_padding(kernel_size, format="full"):
    if kernel_size % 2 == 1:
        top = left = right = bottom = (kernel_size - 1) // 2
    else:
        top = left = (kernel_size - 1) // 2
        bottom = right = (kernel_size - 1) // 2 + 1
    if format == "full":
        return left, right, top, bottom
    elif format == "single" and top == bottom and left == right and top == left:
        return top
    else:
        ValueError("Padding is not symmetric and cannot be represented as a single value")


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the image of shape [B, C, H, W]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


def assert_shape(x, expected_shape):
    assert x.shape == expected_shape, f"Expected shape {expected_shape} but got {x.shape}"


def get_defunct_processes():
    try:
        # Run the ps command and capture the output
        result = subprocess.run(["ps", "-ef"], capture_output=True, text=True, check=True)
        # Filter the output for defunct processes
        defunct_processes = [line for line in result.stdout.splitlines() if "<defunct>" in line]
        return defunct_processes
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return []


def get_defunct_process_pids():
    defunct_processes = get_defunct_processes()
    # Extract the process IDs from the output
    pids = [int(line.split()[1]) for line in defunct_processes]
    return pids


def get_defunct_process_ppids():
    defunct_processes = get_defunct_processes()
    # Extract the parent process IDs from the output
    ppids = [int(line.split()[2]) for line in defunct_processes]
    ppids = list(set(ppids))  # Remove duplicates
    return ppids


def kill_defunct_processes():
    ppids = get_defunct_process_ppids()
    for ppid in ppids:
        try:
            os.kill(ppid, 9)
            print(f"Killed defunct process with PID {ppid}")
        except ProcessLookupError:
            print(f"Process with PID {ppid} not found")


def get_fig_as_array(fig):
    # Save the plot to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    # Convert the BytesIO object to a numpy array
    output_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    output_image = cv2.cvtColor(cv2.imdecode(output_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return output_image


def create_image_grid_seg(images, predictions, labels, grid_size=(4, 4)):
    """
    Create a grid of images with outlines indicating correct and incorrect predictions.

    Args:
    images (tensor): Tensor of images.
    predictions (tensor): Tensor of predictions.
    labels (tensor): Tensor of labels.
    grid_size (tuple): Size of the grid (rows, cols).

    Returns:
    The image grid as an array.
    """
    # Convert tensors to numpy arrays for easier manipulation
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # assuming images are in (N, C, H, W) format
    predictions = predictions.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i]
            pred = predictions[i]
            label = labels[i]

            # Determine the color of the outline
            color = [0, 255, 0] if pred == label else [0, 0, 255]
            canvas = np.full((img.shape[0] + 4, img.shape[1] + 4, 3), color)
            canvas[2:-2, 2:-2] = (img * 255).astype(np.uint8).clip(0, 255)
            ax.imshow(canvas)

        # Remove axis labels
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    grid_image = get_fig_as_array(fig)
    return grid_image


def create_image_grid_denoise(images, predictions, labels, grid_size=(4, 4)):
    """
    Create a grid of images with outlines indicating correct and incorrect predictions.

    Args:
    images (tensor): Tensor of images.
    predictions (tensor): Tensor of predictions.
    labels (tensor): Tensor of labels.
    grid_size (tuple): Size of the grid (rows, cols).

    Returns:
    The image grid as an array.
    """
    # Convert tensors to numpy arrays for easier manipulation
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    predictions = predictions.permute(0, 2, 3, 1).cpu().numpy()
    labels = labels.permute(0, 2, 3, 1).cpu().numpy()

    size = 5
    fig, axes = plt.subplots(grid_size[0], grid_size[1] * 3, figsize=(size * grid_size[0], 3 * size * grid_size[1]))

    for i in range(grid_size[0] * grid_size[1]):
        row = i // grid_size[1]
        col = i % grid_size[1]
        if i < images.shape[0]:
            if col < grid_size[1]:
                axes[row, col * 3].imshow(images[i])
                axes[row, col * 3 + 1].imshow(predictions[i])
                axes[row, col * 3 + 2].imshow(labels[i])

        axes[row, col * 3].axis("off")
        axes[row, col * 3 + 1].axis("off")
        axes[row, col * 3 + 2].axis("off")

    plt.tight_layout()

    grid_image = get_fig_as_array(fig)
    return grid_image


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0  # to [16/255, 240/255]
    return im_ycbcr


def make_deterministic(seed=42):
    L.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = False


pascal_palette = (
    [
        [0, 0, 0],  # Black
        [230, 25, 75],  # Red
        [60, 180, 75],  # Green
        [255, 225, 25],  # Yellow
        [0, 130, 200],  # Blue
        [245, 130, 48],  # Orange
        [145, 30, 180],  # Purple
        [70, 240, 240],  # Cyan
        [240, 50, 230],  # Magenta
        [210, 245, 60],  # Lime
        [250, 190, 190],  # Pink
        [0, 128, 128],  # Teal
        [230, 190, 255],  # Lavender
        [170, 110, 40],  # Brown
        [255, 250, 200],  # Beige
        [128, 0, 0],  # Maroon
        [170, 255, 195],  # Mint
        [128, 128, 0],  # Olive
        [255, 215, 180],  # Coral
        [0, 0, 128],  # Navy
        [128, 128, 128],  # Gray
    ]
    + ([[255, 255, 255]] * 234)
    + [[255, 255, 255]]
)  # White

pascal_palette = torch.tensor(pascal_palette) / 255.0


def create_image_grid_pascal(x, y_hat, y, width=None):
    y_hat_int = y_hat.argmax(dim=1, keepdim=False)
    y_hat_colorized = pascal_palette[y_hat_int].permute(0, 3, 1, 2)
    y_colorized = pascal_palette[y].permute(0, 3, 1, 2)
    lineups = torch.cat([x.cpu(), y_hat_colorized.cpu(), y_colorized.cpu()], dim=3)
    grid = torchvision.utils.make_grid(lineups, nrow=2)
    return grid


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
