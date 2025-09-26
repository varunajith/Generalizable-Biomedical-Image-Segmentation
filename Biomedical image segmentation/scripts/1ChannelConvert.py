# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:20:56 2025

@author: Sayantari
"""

from PIL import Image
import os
from pathlib import Path

def convert_to_grayscale(image_path, output_path):
    """
    Converts an image to grayscale and saves it to the output path.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the grayscale image.
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to grayscale
            grayscale_img = img.convert('L')
            # Save the grayscale image
            grayscale_img.save(output_path)
            print(f"Converted {image_path} to grayscale and saved to {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def convert_folder_to_grayscale(input_folder, output_folder):
    """
    Converts all images in a folder to grayscale and saves them to the output folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save grayscale images.
    """
    # Create the output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif','.tif']

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            # Construct the full file paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Convert the image to grayscale
            convert_to_grayscale(input_path, output_path)

# Example usage
input_folder = "D:/M.Sc_AV_Project/cGAN-Seg-main/cGAN-Seg_datasets/Protein_nanowire_network/train/masks_no_background"  # Replace with your input folder path
output_folder = "D:/M.Sc_AV_Project/cGAN-Seg-main/cGAN-Seg_datasets/Protein_nanowire_network/train/masks_no_background"  # Replace with your output folder path
convert_folder_to_grayscale(input_folder, output_folder)