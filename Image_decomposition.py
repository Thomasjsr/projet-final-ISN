"""
Image Decomposition using OpenCV's Hough Circle Transform.

This script provides functionalities to detect circular objects (presumably coins)
in an input image. It allows configuration of detection parameters via a
command-line interface. Detected circles can be cropped, saved to an output
directory, and displayed. The core detection and cropping logic is encapsulated
in the `detect_and_crop_circles` function, making it reusable by other scripts.

When run as a standalone script, it takes an image path and optional parameters
to perform circle detection and displays the results.
"""
import sys
import cv2 as cv
import numpy as np
import argparse
import os
import logging

# Default target width for resizing the input image before applying HoughCircles.
# This helps in making detection parameters (like min/maxRadius) more consistent
# across images of different original sizes.
DEFAULT_RESIZE_WIDTH = 500

# Default padding (in pixels) to add around the detected circle's radius
# when cropping the image. This ensures the entire object is captured.
DEFAULT_CROP_PADDING = 5


def resize_img(img, target_width=DEFAULT_RESIZE_WIDTH):
    """
    Resizes an image to a specified target width while maintaining its aspect ratio.

    Args:
        img (numpy.ndarray): The input image to resize. Expected to be a valid OpenCV image array.
        target_width (int, optional): The desired width for the resized image.
            Defaults to `DEFAULT_RESIZE_WIDTH`.

    Returns:
        numpy.ndarray: The resized image. Returns `None` if the input image is invalid
                       (e.g., None or has zero width).
    """
    if img is None or img.shape[1] == 0:
        logging.error("Cannot resize_img: Input image is None or has zero width.")
        return None
    # Calculate the ratio of the new width to the old width
    scale_percent = target_width / img.shape[1]
    # Calculate new dimensions
    w = int(img.shape[1] * scale_percent)
    h = int(img.shape[0] * scale_percent)
    dim = (w, h)
    # Resize the image using cubic interpolation for better quality
    new_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    logging.debug(f"Resized image from {img.shape[1]}x{img.shape[0]} to {w}x{h}")
    return new_img


def detect_and_crop_circles(input_image_arr, param1, param2, min_radius, max_radius,
                            resize_target_width=DEFAULT_RESIZE_WIDTH, crop_padding=DEFAULT_CROP_PADDING):
    """
    Detects circles in an image using Hough Circle Transform, crops them,
    and returns the cropped images along with their coordinates and the image
    with detected circles drawn on it.

    The input image is first resized (maintaining aspect ratio) to `resize_target_width`
    for more consistent circle detection.

    Args:
        input_image_arr (numpy.ndarray): The input image as a NumPy array (e.g., loaded by cv2.imread).
        param1 (int): First parameter for `cv.HoughCircles`. It is the higher threshold for the Canny edge detector.
        param2 (int): Second parameter (accumulator threshold) for `cv.HoughCircles`.
                      The smaller it is, the more false circles may be detected.
        min_radius (int): Minimum circle radius (in pixels) to detect.
        max_radius (int): Maximum circle radius (in pixels) to detect.
        resize_target_width (int, optional): Width to resize the input image to before circle detection.
                                             Defaults to `DEFAULT_RESIZE_WIDTH`.
        crop_padding (int, optional): Padding (in pixels) to add around the detected circle's radius
                                      when cropping. Defaults to `DEFAULT_CROP_PADDING`.

    Returns:
        tuple: A tuple `(list_of_cropped_data, image_with_circles_drawn)` where:
               - `list_of_cropped_data` (list): A list of tuples. Each tuple contains:
                 `(cropped_image_array, center_coordinates_tuple, radius_integer)`.
                 Coordinates and radius are relative to the `resize_target_width` image.
               - `image_with_circles_drawn` (numpy.ndarray): The (resized) image with detected circles drawn on it.
                 This is `None` if the input image was invalid.
               Returns `([], input_image_arr)` if `input_image_arr` is `None` or if resize fails.
               Returns `([], image_with_circles_drawn)` if no circles are found.
    """
    if input_image_arr is None:
        logging.error("Input image array is None in detect_and_crop_circles.")
        return [], None # Return empty list and None for the display image

    logging.info(f"Resizing image to width {resize_target_width} for circle detection.")
    # Resize the image for more consistent circle detection
    processed_img = resize_img(input_image_arr, target_width=resize_target_width)
    if processed_img is None:
        logging.error("Failed to resize image for circle detection.")
        # Return the original image in this case, as processed_img is None
        return [], input_image_arr

    # Convert to grayscale for HoughCircles
    gray = cv.cvtColor(processed_img, cv.COLOR_BGR2GRAY)
    # Apply median blur to reduce noise and improve detection
    gray = cv.medianBlur(gray, 5)
    logging.info("Image preprocessed (grayscale, median blur). Detecting circles...")

    rows = gray.shape[0] # Number of rows in the processed image
    # Detect circles. dp=1 (inverse ratio of accumulator resolution),
    # minDist=rows/8 (minimum distance between centers of detected circles).
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=rows / 8,
                              param1=param1, param2=param2,
                              minRadius=min_radius, maxRadius=max_radius)

    image_with_circles_drawn = processed_img.copy()
    cropped_coin_data = [] # To store (cropped_array, center, radius)

    if circles is not None:
        logging.info(f"Detected {circles.shape[1]} potential circles.")
        # Convert circle parameters (x, y, radius) to integers
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]: # Iterate through detected circles
            center = (i[0], i[1]) # Circle center (x, y)
            radius = i[2]        # Circle radius
            logging.debug(f"Processing circle at center: {center}, radius: {radius}")

            # Draw visualizations on the copy of the processed image
            cv.circle(image_with_circles_drawn, center, 1, (0, 100, 100), 3)    # Small dot at center
            cv.circle(image_with_circles_drawn, center, radius, (255, 0, 255), 3) # Circle outline

            # Calculate cropping box coordinates with padding
            # Ensure coordinates are within image bounds
            y_start = max(0, center[1] - radius - crop_padding)
            y_end = min(processed_img.shape[0], center[1] + radius + crop_padding)
            x_start = max(0, center[0] - radius - crop_padding)
            x_end = min(processed_img.shape[1], center[0] + radius + crop_padding)
            
            # Crop the image
            img_cropped = processed_img[y_start:y_end, x_start:x_end]

            if img_cropped.size == 0: # Check if the cropped image is empty
                logging.warning(f"Cropped image for circle at {center} with radius {radius} is empty. Skipping.")
                continue
            
            cropped_coin_data.append((img_cropped, center, radius))
        logging.info(f"Extracted {len(cropped_coin_data)} valid cropped images.")
    else:
        logging.info("No circles detected.")
    
    return cropped_coin_data, image_with_circles_drawn


def main_cli(args):
    """
    Main function for the command-line interface (CLI) of this script.
    Loads an image specified by CLI arguments, detects circles using parameters
    from CLI, and then saves or displays the results based on CLI flags.

    Args:
        args (argparse.Namespace): Parsed command-line arguments. Expected to contain:
            - input_image_path (str): Path to the input image.
            - param1 (int): HoughCircles param1.
            - param2 (int): HoughCircles param2.
            - minRadius (int): HoughCircles minRadius.
            - maxRadius (int): HoughCircles maxRadius.
            - output_dir (str, optional): Directory to save cropped images.
            - no_display_cropped (bool): If True, suppresses display of individual cropped images.
    """
    logging.info(f"Starting image processing for: {args.input_image_path}")
    # Load the source image using OpenCV
    # cv.samples.findFile is typically for OpenCV's own samples, direct path is better here.
    src_image = cv.imread(args.input_image_path, cv.IMREAD_COLOR)

    if src_image is None:
        logging.error(f'Error opening image: {args.input_image_path}. Ensure it exists and is valid.')
        return -1 # Indicate error

    # Perform circle detection and cropping using the core function
    cropped_coin_data, image_with_circles = detect_and_crop_circles(
        src_image,
        args.param1, args.param2, args.minRadius, args.maxRadius,
        # These could also be made CLI arguments if desired for more flexibility
        resize_target_width=DEFAULT_RESIZE_WIDTH, 
        crop_padding=DEFAULT_CROP_PADDING
    )

    if not cropped_coin_data:
        logging.info("No coins were cropped from the image.")
    
    # Prepare for saving: get base filename and extension
    base_filename, file_ext = os.path.splitext(os.path.basename(args.input_image_path))
    num_saved = 0
    for idx, (cropped_img_arr, center, radius) in enumerate(cropped_coin_data):
        # Save cropped image if an output directory is specified
        if args.output_dir:
            try:
                os.makedirs(args.output_dir, exist_ok=True) # Create output directory if it doesn't exist
                cropped_filename = f"{base_filename}_coin_{idx}{file_ext}"
                save_path = os.path.join(args.output_dir, cropped_filename)
                cv.imwrite(save_path, cropped_img_arr)
                logging.info(f"Saved cropped image: {save_path}")
                num_saved += 1
            except Exception as e:
                logging.error(f"Error saving image {save_path}: {e}")

        # Display individual cropped images unless suppressed by --no_display_cropped
        if not args.no_display_cropped:
            window_name = f"Cropped Coin {idx} (Center: {center}, R: {radius})"
            cv.imshow(window_name, cropped_img_arr)
    
    if args.output_dir:
        logging.info(f"Saved {num_saved} cropped images to {args.output_dir}")

    # Display the (resized) image with all detected circles drawn on it
    # Ensure image_with_circles is not None before trying to show it
    if image_with_circles is not None:
        cv.imshow("Detected Circles on Resized Image", image_with_circles)
    else:
        logging.warning("Image with circles to display is None (e.g. input image was invalid).")
        
    cv.waitKey(0) # Wait for a key press to close displayed windows
    cv.destroyAllWindows() # Close all OpenCV windows
    logging.info("Image processing finished.")
    return 0 # Indicate successful execution


if __name__ == "__main__":
    # Configure basic logging for the script.
    # This will show messages of level INFO and above, with a timestamp.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Detects circles (e.g., coins) in an image using OpenCV's Hough Transform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument("input_image_path", nargs='?', default='test.jpg',
                        help="Path to the input image.")
    # Hough Circle Transform parameters
    parser.add_argument("--param1", type=int, default=100,
                        help="First parameter for HoughCircles (gradient value for Canny edge detection).")
    parser.add_argument("--param2", type=int, default=30,
                        help="Second parameter for HoughCircles (accumulator threshold for circle centers).")
    parser.add_argument("--minRadius", type=int, default=1,
                        help="Minimum circle radius to detect (pixels).")
    parser.add_argument("--maxRadius", type=int, default=70,
                        help="Maximum circle radius to detect (pixels).")
    # Output options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save cropped coin images. If not specified, images are not saved.")
    parser.add_argument("--no_display_cropped", action="store_true",
                        help="If set, suppresses the display of individual cropped coin images.")

    cli_args = parser.parse_args()
    sys.exit(main_cli(cli_args))
