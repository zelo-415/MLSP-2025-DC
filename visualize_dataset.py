import torch
import matplotlib.pyplot as plt
from dataset import RadioMapDataset
from pathlib import Path
import numpy as np

def visualize_dataset(dataset, index, output_dir):
    """
    Saves the visualizations of the input tensor, ground truth tensor, and mask map for a given index in the dataset.
    Also writes the transmitter (tx_x, tx_y) coordinates on the images for context.

    Args:
        dataset (RadioMapDataset): The dataset object.
        index (int): The index of the sample to visualize.
        output_dir (str): Directory to save the visualizations.
    """
    input_tensor, gt_tensor, mask_map = dataset[index]

    # Convert tensors to numpy arrays for visualization
    input_np = input_tensor.numpy()
    gt_np = gt_tensor.squeeze().numpy()
    mask_np = mask_map.squeeze().numpy()

    # Get the transmitter coordinates
    fname = dataset.filenames[index]
    
    tx_x, tx_y = dataset._load_tx_xy(fname)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all input tensor channels in one image
    num_channels = input_np.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
    for i in range(num_channels):
        axes[i].imshow(input_np[i], cmap="viridis")
        axes[i].set_title(f"Input Channel {i+1}")
        axes[i].axis("off")
        # Annotate the transmitter coordinates
        axes[i].text(10, 10, f"Tx: ({tx_x:.1f}, {tx_y:.1f})", color="white", fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f"input_channels_index_{index}.png", bbox_inches="tight")
    plt.close()

    # Save the ground truth tensor
    plt.figure(figsize=(6, 6))
    plt.imshow(gt_np, cmap="hot")
    plt.title("Ground Truth Tensor")
    plt.colorbar(label="PL Value")
    plt.axis("off")
    plt.text(10, 10, f"Tx: ({tx_x:.1f}, {tx_y:.1f})", color="white", fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    plt.savefig(output_dir / f"ground_truth_index_{index}.png", bbox_inches="tight")
    plt.close()

    # Save the mask map
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_np, cmap="gray")
    plt.title("Mask Map")
    plt.axis("off")
    plt.text(10, 10, f"Tx: ({tx_x:.1f}, {tx_y:.1f})", color="white", fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    plt.savefig(output_dir / f"mask_map_index_{index}.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Define dataset paths
    inputs_dir = "inputs"
    outputs_dir = "outputs"
    sparse_dir = "sparse_samples_0.5"
    positions_dir = "Positions"
    los_dir = "losmap"
    hit_dir = "hitmap"

    # Create the dataset
    dataset = RadioMapDataset(inputs_dir, outputs_dir, sparse_dir, positions_dir, los_dir, hit_dir)

    # Define output directory for visualizations
    output_dir = "visualizations"

    # Visualize and save a sample from the dataset
    sample_index = 700  # Change this index to visualize other samples
    visualize_dataset(dataset, sample_index, output_dir)