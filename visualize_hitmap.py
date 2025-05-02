import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_hitmap(hitmap_dir, output_dir=None):
    """
    Visualizes all .npy files in the hitmap directory.

    Args:
        hitmap_dir (str): Path to the directory containing hitmap .npy files.
        output_dir (str, optional): Path to save the visualizations as images. If None, displays them.
    """
    hitmap_dir = Path(hitmap_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    count = -1
    for hitmap_file in sorted(hitmap_dir.glob("*.npy")):
        if count > 10:
            break
        count += 1
        try:
            # Load the hitmap
            hitmap = np.load(hitmap_file)

            # Plot the hitmap
            plt.figure(figsize=(8, 6))
            plt.imshow(hitmap, cmap="hot", interpolation="nearest")
            plt.colorbar(label="Hit Intensity")
            plt.title(f"Hit Map: {hitmap_file.stem}")
            plt.axis("off")

            if output_dir:
                # Save the visualization
                output_path = output_dir / f"{hitmap_file.stem}.png"
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()
            else:
                # Display the visualization
                plt.show()

        except Exception as e:
            print(f"[ERROR] Could not visualize {hitmap_file.name}: {e}")

if __name__ == "__main__":
    hitmap_dir = "hitmap"  # Path to the hitmap folder
    output_dir = "./"  # Set to a folder path to save visualizations, or None to display them
    visualize_hitmap(hitmap_dir, output_dir)