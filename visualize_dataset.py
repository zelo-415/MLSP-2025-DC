import torch
import matplotlib.pyplot as plt
from dataset import RadioMapDataset
from pathlib import Path

def visualize_dataset(dataset, index, output_dir):
    """
    Saves the visualizations of the input tensor, ground truth tensor, and mask map for a given index in the dataset.

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

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the input tensor channels
    num_channels = input_np.shape[0]
    for i in range(num_channels):
        plt.figure(figsize=(6, 6))
        plt.imshow(input_np[i], cmap="viridis")
        plt.title(f"Input Channel {i+1}")
        plt.axis("off")
        plt.savefig(output_dir / f"input_channel_{i+1}_index_{index}.png", bbox_inches="tight")
        plt.close()

    # Save the ground truth tensor
    plt.figure(figsize=(6, 6))
    plt.imshow(gt_np, cmap="hot")
    plt.title("Ground Truth Tensor")
    plt.colorbar(label="PL Value")
    plt.axis("off")
    plt.savefig(output_dir / f"ground_truth_index_{index}.png", bbox_inches="tight")
    plt.close()

    # Save the mask map
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_np, cmap="gray")
    plt.title("Mask Map")
    plt.axis("off")
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
    sample_index = 0  # Change this index to visualize other samples
    visualize_dataset(dataset, sample_index, output_dir)