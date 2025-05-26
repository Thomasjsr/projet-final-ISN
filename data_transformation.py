"""
Data Transformation Script for Image Pixel Data Extraction.

This script is designed to process directories of images, organized by category
(e.g., different types of coins). For each image, it extracts pixel color data
and associates it with a numerical label corresponding to its category.
The extracted data (pixel color, label) is then saved into text files,
with one text file generated per image.

The primary function, `preprocess_image_data`, handles the iteration through
specified subdirectories (typically 'train' and 'test' sets) and categories,
reading images, and writing their pixel data to output files.

Command-line arguments allow specification of the base data directory,
the list of categories, and optionally, a specific subdirectory (train/test)
to process. If no subdirectory is specified, both 'train' and 'test' are processed.

The output text files are named based on the original image name, its category,
and the subdirectory it belongs to (e.g., `image1_1c_train.txt`). These files
are saved in the current working directory where the script is executed.
The `file_handler.py` module was previously used for file writing but has been
integrated directly into this script for simplicity in this specific use case.
"""
import os
from PIL import Image # Pillow library for image manipulation
import file_handler # Though not used in preprocess_image_data after refactor, kept for other potential uses
import argparse
import logging
import sys

def preprocess_image_data(base_data_dir, categories, subdir):
    """
    Processes images in a specified subdirectory (e.g., 'train' or 'test')
    by iterating through category subfolders. For each image, it extracts
    pixel data (color) and its category label, saving this information
    into a new text file specific to that image.

    Args:
        base_data_dir (str): The root directory containing the dataset structure
                             (e.g., `coins_dataset-master/classified`).
        categories (list[str]): A list of category names. Each name should correspond
                                to a subfolder within `base_data_dir/subdir/`.
        subdir (str): The subdirectory to process (typically "train" or "test").

    Returns:
        None. Output is written to files.
    """
    # Construct the full path to the current subdirectory (e.g., .../classified/train)
    datadir = os.path.join(base_data_dir, subdir)
    logging.info(f"Starting preprocessing for subdirectory: {datadir}")
    processed_image_count = 0 # Counter for successfully processed images

    # Iterate over each specified category
    for category_name in categories:
        category_path = os.path.join(datadir, category_name)
        # Assign a numerical label to the category based on its index in the list
        label = categories.index(category_name)

        # Check if the category path exists
        if not os.path.exists(category_path):
            logging.warning(f"Category path not found, skipping: {category_path}")
            continue # Move to the next category
        
        try:
            # List all files (images) in the current category directory
            image_names = os.listdir(category_path)
            if not image_names:
                logging.warning(f"No images found in category directory: {category_path}")
                continue # Move to the next category
        except FileNotFoundError: # Safety check, though os.path.exists should catch this
            logging.error(f"Directory not found during listdir: {category_path}. This should have been caught earlier.")
            continue
        except Exception as e:
            logging.error(f"Error listing directory contents for {category_path}: {e}")
            continue

        logging.info(f"Processing category: '{category_name}' in subdirectory '{subdir}'")
        # Iterate over each image in the category
        for img_name in image_names:
            img_path = os.path.join(category_path, img_name)
            try:
                logging.debug(f"Opening image: {img_path}")
                # Open the image using Pillow
                image = Image.open(img_path)
                w, h = image.size # Get image dimensions
                
                # Define the output filename for the extracted pixel data.
                # Format: original_name_category_subdir.txt
                # Example: "image01_1c_train.txt"
                # Files are saved in the script's current working directory.
                output_filename = f"{os.path.splitext(img_name)[0]}_{category_name}_{subdir}.txt"
                
                logging.info(f'Creating data file: "{output_filename}" for image "{img_name}"')
                # Open the output file in write mode ('w')
                with open(output_filename, 'w') as f:
                    # Iterate over each pixel in the image
                    for x in range(w):
                        for y in range(h):
                            try:
                                # Get the color of the pixel at (x, y)
                                color = image.getpixel((x, y))
                                # Format the data line: (pixel_color, label)
                                data_line = str((color, label)) + "\n"
                                f.write(data_line)
                            except Exception as e_pixel:
                                logging.warning(f"Error getting pixel ({x},{y}) from {img_path}: {e_pixel}")
                processed_image_count += 1 # Increment count of processed images
            except IOError as e_io: # Specific error for file I/O issues (e.g., cannot open image)
                logging.error(f"Error opening or reading image {img_path}: {e_io}")
            except Exception as e_general: # Catch any other errors during image processing
                logging.error(f"General error processing image {img_path}: {e_general}")
    
    logging.info(f"Finished preprocessing for subdirectory '{subdir}'. Processed {processed_image_count} images.")


if __name__ == "__main__":
    # Configure basic logging settings
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Preprocesses image data by extracting pixel information and corresponding labels. "
                    "Saves output to text files in the current working directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help messages
    )
    parser.add_argument('--base_dir', type=str, default="coins_dataset-master/classified", 
                        help="Base directory containing 'train' and 'test' subdirectories, which in turn contain category subfolders.")
    parser.add_argument('--categories', type=str, nargs='+', 
                        default=["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"],
                        help="List of category names. These should match subfolder names within the 'train' and 'test' directories.")
    parser.add_argument('--subdir', type=str, 
                        help="Specific subdirectory to process (e.g., 'train' or 'test'). "
                             "If not specified, both 'train' and 'test' subdirectories will be processed.")
    
    args = parser.parse_args() # Parse the command-line arguments

    subdirs_to_process = []
    if args.subdir: # If a specific subdirectory is provided
        if args.subdir not in ["train", "test"]: # Validate the input
            logging.error(f"Invalid --subdir argument: '{args.subdir}'. Must be 'train' or 'test'.")
            sys.exit(1) # Exit with an error code
        subdirs_to_process.append(args.subdir)
    else: # If no specific subdirectory is given, process both 'train' and 'test'
        subdirs_to_process.extend(["train", "test"])

    # Loop through the selected subdirectories and process them
    for subdir_name in subdirs_to_process:
        logging.info(f"--- Starting processing for subdirectory: {subdir_name} ---")
        preprocess_image_data(args.base_dir, args.categories, subdir_name)
    
    logging.info("All preprocessing tasks completed.")
