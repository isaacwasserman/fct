import torch
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import PIL.Image
import cv2


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
    pass
    # print("Asserting shape", x.shape, expected_shape)
    # assert x.shape == expected_shape, f"Expected shape {expected_shape} but got {x.shape}"


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


def create_image_grid(images, predictions, labels, grid_size=(4, 4)):
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
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    # Convert the BytesIO object to a numpy array
    grid_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    grid_image = cv2.imdecode(grid_image, cv2.IMREAD_COLOR)
    return grid_image
