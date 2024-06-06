import torch
import subprocess
import os


def same_padding(kernel_size):
    if kernel_size % 2 == 1:
        top = left = right = bottom = (kernel_size - 1) // 2
    else:
        top = left = (kernel_size - 1) // 2
        bottom = right = (kernel_size - 1) // 2 + 1
    return left, right, top, bottom


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
