"""
Coin Classification Pipeline Script.

This script orchestrates a multi-step process to identify and classify coins
within a given input image. It leverages functionalities from two other modules:
1.  `Image_decomposition.py`: For detecting circular objects (potential coins)
    in the image using Hough Circle Transform and cropping them.
2.  `coin_detector.py`: For training a K-Nearest Neighbors (KNN) model using
    a predefined dataset of coin images and then using this model to classify
    the detected coin candidates.

The pipeline involves:
-   Training the KNN model on-the-fly using data specified in `coin_detector.py`.
    (Note: Model persistence, i.e., saving/loading a pre-trained model, is
    currently not implemented).
-   Loading the user-provided input image.
-   Applying circle detection to find potential coins in the input image.
-   For each detected circular region (cropped coin):
    -   Preprocessing the cropped image (grayscale conversion, resizing to match
      the KNN model's expected input size, flattening into a feature vector).
    -   Classifying the preprocessed coin using the trained KNN model.
    -   Logging the classification result (category of the coin).
-   Optionally saving the classified cropped coin images to an output directory.
-   Displaying the input image with detected circles marked.

Command-line arguments allow customization of image paths, detection parameters,
and output behavior.
"""
import cv2
import numpy as np
import os
import logging
import argparse
import sys

# Attempt to import necessary functions and constants from local sibling modules.
# These modules should be in the same directory or accessible via PYTHONPATH.
try:
    from Image_decomposition import (detect_and_crop_circles,
                                     DEFAULT_RESIZE_WIDTH as ID_DEFAULT_RESIZE_WIDTH,
                                     DEFAULT_CROP_PADDING)
    from coin_detector import train_knn_model, CATEGORIES, TARGET_IMG_SIZE
except ImportError as e:
    logging.error(f"Critical Error: Failed to import utility modules (Image_decomposition or coin_detector). "
                  f"Ensure they are in the Python path. Details: {e}")
    sys.exit(1) # Exit if essential modules cannot be imported.

def preprocess_cropped_coin_for_knn(cropped_image_arr, target_size=TARGET_IMG_SIZE):
    """
    Preprocesses a single cropped coin image to prepare it for KNN classification.

    The preprocessing steps are:
    1.  Convert the image to grayscale if it's not already.
    2.  Resize the image to the `target_size` (this size must match the dimensions
        of images used for training the KNN model).
    3.  Flatten the 2D image array into a 1D feature vector.

    Args:
        cropped_image_arr (numpy.ndarray): The cropped coin image (a 2D or 3D NumPy array).
        target_size (tuple[int, int], optional): The target (width, height) to resize to.
                                                 Defaults to `TARGET_IMG_SIZE` imported from `coin_detector`.

    Returns:
        numpy.ndarray: The flattened 1D feature vector ready for KNN classification.
                       Returns `None` if the input image is invalid or if any
                       preprocessing step fails.
    """
    if cropped_image_arr is None or cropped_image_arr.size == 0:
        logging.warning("Cannot preprocess: Input cropped_image_arr is None or empty.")
        return None
    
    try:
        # Ensure image is grayscale
        if cropped_image_arr.ndim == 3 and cropped_image_arr.shape[2] == 3: # Check for 3 channels (color)
            gray_coin = cv2.cvtColor(cropped_image_arr, cv2.COLOR_BGR2GRAY)
        elif cropped_image_arr.ndim == 2: # Already grayscale
            gray_coin = cropped_image_arr
        else:
            logging.warning(f"Cropped image has an unexpected number of dimensions/channels: {cropped_image_arr.shape}. "
                            "Cannot reliably convert to grayscale.")
            return None

        # Resize to the target size expected by the KNN model
        resized_coin = cv2.resize(gray_coin, target_size)
        # Flatten the 2D image array into a 1D feature vector
        flattened_coin = resized_coin.flatten()
        logging.debug(f"Preprocessed cropped coin. Original_shape (cropped): {cropped_image_arr.shape}, "
                      f"Resized_shape: {resized_coin.shape}, Flattened_shape: {flattened_coin.shape}")
        return flattened_coin
    except Exception as e:
        logging.error(f"Error during cropped coin preprocessing (resize/flatten): {e}")
        return None

def main_classification_pipeline(args):
    """
    Executes the main coin detection and classification pipeline.

    This involves training a KNN model, loading an input image, detecting circles
    (coins) in it, preprocessing each detected coin, classifying it using the
    trained model, and then logging/saving the results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing paths,
                                   detection parameters, and output options.

    Returns:
        int: Exit code (0 for success, -1 for critical failure).
    """
    logging.info("Starting coin classification pipeline...")

    # --- Step 1: Train KNN Model ---
    # The KNN model is trained on-the-fly each time the script runs.
    # TARGET_IMG_SIZE from coin_detector is crucial here, as it defines the
    # dimensions training images are resized to, and thus the expected input
    # size for classification.
    logging.info("Training KNN model (on-the-fly)...")
    knn_model = train_knn_model() # Uses defaults from coin_detector.py
    if knn_model is None:
        logging.error("Critical: KNN model training failed. Cannot proceed with classification. Exiting pipeline.")
        return -1 # Indicate critical failure
    logging.info("KNN model trained successfully.")

    # --- Step 2: Load Input Image and Perform Circle Detection ---
    logging.info(f"Loading input image from: {args.input_image_path}")
    # cv.samples.findFile is for OpenCV's own sample images, not general paths.
    input_image = cv2.imread(args.input_image_path)
    if input_image is None:
        logging.error(f"Failed to load input image: '{args.input_image_path}'. Please check the path and file integrity.")
        return -1
    
    logging.info("Detecting and cropping circles (potential coins) from the input image...")
    # Call the function from Image_decomposition.py.
    # This function handles its own image resizing for optimal circle detection.
    cropped_coin_data_list, image_with_circles_drawn = detect_and_crop_circles(
        input_image_arr=input_image,
        param1=args.param1,
        param2=args.param2,
        min_radius=args.minRadius,
        max_radius=args.maxRadius,
        resize_target_width=args.detection_resize_width, # Width for circle detection stage
        crop_padding=args.crop_padding
    )

    if not cropped_coin_data_list:
        logging.info("No circles (potential coins) were detected or extracted from the input image.")
        # Display the image (it might have no circles drawn if none were found by detect_and_crop_circles)
        if image_with_circles_drawn is not None:
            cv2.imshow("Input Image (No Coins Detected/Cropped)", image_with_circles_drawn)
        else: # Should not happen if input_image was valid, but as a fallback
            cv2.imshow("Input Image (Original, Error in Detection Step)", input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0 # Successful execution, though no coins to classify

    logging.info(f"Successfully detected and cropped {len(cropped_coin_data_list)} potential coin regions.")

    # --- Step 3: Coin Preprocessing and Classification ---
    classified_coin_count = 0
    # Create a copy of the original image if we want to draw final classification results on it.
    # image_for_final_classification_drawing = input_image.copy() 

    for idx, (cropped_img_arr, center_on_resized_img, radius_on_resized_img) in enumerate(cropped_coin_data_list):
        logging.debug(f"Processing cropped coin candidate #{idx} obtained from the (detection-resized) image.")
        
        # Preprocess the cropped coin image for KNN classification.
        # This includes resizing to TARGET_IMG_SIZE (used during KNN training) and flattening.
        flattened_coin_vector = preprocess_cropped_coin_for_knn(cropped_img_arr, TARGET_IMG_SIZE)

        if flattened_coin_vector is None:
            logging.warning(f"Skipping classification for coin candidate #{idx} due to preprocessing error.")
            continue # Move to the next cropped image

        # Classify the preprocessed coin using the trained KNN model.
        # The KNN model's predict method expects a list/array of samples.
        try:
            # The predict method should return a list of predictions, one for each input sample.
            prediction_index_list = knn_model.predict([flattened_coin_vector]) 
            
            if not prediction_index_list: # Should not occur if predict works as expected
                logging.error(f"Prediction for coin candidate #{idx} returned an empty list.")
                continue
            
            predicted_class_index = prediction_index_list[0] # Get the first (and only) prediction
            # Map the predicted class index to a category name using CATEGORIES from coin_detector.
            predicted_category_name = CATEGORIES[predicted_class_index]
            classified_coin_count += 1
            
            logging.info(f"Coin Candidate #{idx}: "
                         f"Details (on detection-resized image) -> Center: {center_on_resized_img}, Radius: {radius_on_resized_img} "
                         f"==> Classified as: {predicted_category_name}")

            # TODO (Optional Enhancement): Map `center_on_resized_img` and `radius_on_resized_img`
            # back to the original input_image's coordinates if results need to be drawn on it.
            # This requires the resize factor used within `detect_and_crop_circles`.
            # Example:
            # scale_factor = input_image.shape[1] / args.detection_resize_width
            # original_center = (int(center_on_resized_img[0] * scale_factor), int(center_on_resized_img[1] * scale_factor))
            # original_radius = int(radius_on_resized_img * scale_factor)
            # cv2.putText(image_for_final_classification_drawing, predicted_category_name, 
            #             (original_center[0] - original_radius, original_center[1] - original_radius - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.circle(image_for_final_classification_drawing, original_center, original_radius, (255, 0, 0), 2)

            # Save the original color cropped image (cropped_img_arr from detect_and_crop_circles)
            # with classification information in the filename, if an output directory is specified.
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True) # Ensure output directory exists
                original_basename, original_ext = os.path.splitext(os.path.basename(args.input_image_path))
                
                save_filename = f"{original_basename}_coin_{idx}_as_{predicted_category_name}{original_ext}"
                save_path = os.path.join(args.output_dir, save_filename)
                try:
                    cv2.imwrite(save_path, cropped_img_arr) # Save the color cropped image
                    logging.info(f"Saved classified coin image to: {save_path}")
                except Exception as e:
                    logging.error(f"Error saving cropped coin image {save_path}: {e}")

        except IndexError: # If predicted_class_index is out of bounds for CATEGORIES
             logging.error(f"Predicted class index {predicted_class_index} is out of range for CATEGORIES list.")
        except Exception as e:
            logging.error(f"Error during classification or result logging for coin candidate #{idx}: {e}")

    logging.info(f"Classification phase complete. Successfully classified {classified_coin_count} "
                 f"out of {len(cropped_coin_data_list)} detected coin candidates.")

    # Display the image that had circles drawn on it during the detection phase.
    # This image is typically a resized version of the original input.
    if image_with_circles_drawn is not None:
        cv2.imshow("Input Image with Detected Circles (Resized for Detection)", image_with_circles_drawn)
    # If drawing on the original full-size image was implemented:
    # cv2.imshow("Input Image with Classifications (Original Size)", image_for_final_classification_drawing)
    
    cv2.waitKey(0) # Wait for a key press
    cv2.destroyAllWindows() # Close all OpenCV windows
    logging.info("Coin classification pipeline finished.")
    return 0 # Indicate successful execution


if __name__ == "__main__":
    # Configure basic logging for the script.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Detects and classifies coins in an input image using a KNN model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument("input_image_path", help="Path to the input image file.")
    
    # Parameters for the circle detection phase (passed to Image_decomposition.detect_and_crop_circles)
    parser.add_argument("--detection_resize_width", type=int, default=ID_DEFAULT_RESIZE_WIDTH,
                        help="Width to resize the input image to specifically for the circle detection phase.")
    parser.add_argument("--crop_padding", type=int, default=DEFAULT_CROP_PADDING,
                        help="Padding (in pixels) to add around a detected circle when cropping it.")
    parser.add_argument("--param1", type=int, default=100,
                        help="HoughCircles: First parameter (higher Canny edge threshold).")
    parser.add_argument("--param2", type=int, default=30,
                        help="HoughCircles: Second parameter (accumulator threshold for circle centers).")
    parser.add_argument("--minRadius", type=int, default=1,
                        help="HoughCircles: Minimum circle radius to detect (pixels).")
    parser.add_argument("--maxRadius", type=int, default=70, 
                        help="HoughCircles: Maximum circle radius to detect (pixels).")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional. Directory to save images of classified cropped coins. "
                             "If not specified, images are not saved.")
    
    # Note: --model_path argument is deferred as the KNN class in AI.KNN_classifier.py
    # currently does not support saving/loading models. The model is trained on-the-fly.
    # Note: --min_confidence argument is deferred as the KNN class does not provide
    # classification confidence scores.

    args = parser.parse_args() # Parse the command-line arguments
    
    # Critical check: Ensure TARGET_IMG_SIZE (for KNN preprocessing) is available from coin_detector.
    # This should have been imported, but an explicit check guards against import issues.
    if 'TARGET_IMG_SIZE' not in globals() or TARGET_IMG_SIZE is None:
        logging.error("Critical: TARGET_IMG_SIZE (for KNN image preprocessing) not found or is None. "
                      "This constant should be imported from coin_detector.py. Cannot proceed.")
        sys.exit(1)

    # Execute the main pipeline
    result = main_classification_pipeline(args)
    sys.exit(result) # Exit with the result code from the pipeline
