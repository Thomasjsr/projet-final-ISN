"""
Coin Detector and KNN Model Trainer.

This script serves two primary purposes:
1.  Provides functions to load image data for coins, preprocess it (including
    resizing to a fixed size and flattening), and train a K-Nearest Neighbors
    (KNN) classifier. The trained model can then be used by other scripts for
    coin classification (e.g., `classify_coins_in_image.py`).
2.  When run as a standalone script, it trains a KNN model using data from the
    `BASE_DATA_DIR` ('classified' by default) and then evaluates the model's
    accuracy on a test set, printing the accuracy to the console.

The KNN classifier itself is imported from `AI.KNN_classifier`.
Image data is expected to be organized into subdirectories named by category
(e.g., "1c", "2c") within "train" and "test" folders under `BASE_DATA_DIR`.
"""
import numpy as np
import os
import cv2
from sklearn.metrics import accuracy_score
from AI.KNN_classifier import KNN # Import custom KNN classifier
import logging
import sys

# Global constants defining the data structure and preprocessing parameters.
# CATEGORIES: List of coin types, corresponding to subdirectory names.
CATEGORIES = ["1c", "2c", "5c", "10c", "20c", "50c", "1e", "2e"]
# BASE_DATA_DIR: Root directory containing 'train' and 'test' subfolders for coin images.
BASE_DATA_DIR = "classified"
# TARGET_IMG_SIZE: Tuple (width, height) to which all images are resized before processing.
# This ensures consistent feature vector sizes for the KNN model.
TARGET_IMG_SIZE = (64, 64)

def load_and_preprocess_data(data_dir, subset, categories_list, target_size=TARGET_IMG_SIZE):
    """
    Loads images from a specified path, resizes them to a consistent `target_size`,
    converts them to grayscale, and organizes them into lists of image arrays and labels.

    This function is used for loading both training and testing datasets.

    Args:
        data_dir (str): The base directory where the data (e.g., 'classified') is located.
        subset (str): The type of data subset to load, typically "train" or "test".
                      This corresponds to a subdirectory within `data_dir`.
        categories_list (list[str]): A list of category names. Each category name should
                                     correspond to a subdirectory within `data_dir/subset/`.
        target_size (tuple[int, int], optional): The target (width, height) to resize images to.
                                                 Defaults to `TARGET_IMG_SIZE`.

    Returns:
        tuple[list[np.ndarray], list[int]]: A tuple containing two lists:
            - `x_data`: A list of image arrays (NumPy arrays), each resized and grayscaled.
            - `y_data`: A list of corresponding integer labels (class indices).
            Returns empty lists if no data is found or errors occur.
    """
    x_data = []
    y_data = []
    logging.info(f"Loading '{subset}' data from '{os.path.join(data_dir, subset)}'...")
    for category_name in categories_list:
        # Construct path to the category's image folder
        path = os.path.join(data_dir, subset, category_name)
        class_num = categories_list.index(category_name) # Get numerical label for the category

        if not os.path.exists(path):
            logging.warning(f"Data path not found for category '{category_name}' in '{subset}' set: {path}")
            continue # Skip to next category
        
        image_files = os.listdir(path)
        if not image_files:
            logging.warning(f"No images found in directory: {path}")
            continue # Skip to next category

        logging.debug(f"Processing {len(image_files)} images in {path}...")
        for img_name in image_files:
            try:
                img_path = os.path.join(path, img_name)
                # Read image in grayscale mode
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is not None:
                    # Resize image to the defined target size
                    resized_img = cv2.resize(img_array, target_size)
                    x_data.append(resized_img)
                    y_data.append(class_num)
                else:
                    logging.warning(f"Could not read image (it might be corrupted or not an image): {img_path}")
            except Exception as e:
                logging.error(f"Error loading or resizing image {img_path}: {e}")
    logging.info(f"Loaded and preprocessed {len(x_data)} images for '{subset}' set. Target image size: {target_size}.")
    return x_data, y_data

def train_knn_model(base_data_dir=BASE_DATA_DIR, categories=CATEGORIES, target_size=TARGET_IMG_SIZE):
    """
    Loads and preprocesses training data, then uses it to train a KNN model.

    Preprocessing steps include resizing images to `target_size` and flattening them
    into 1D feature vectors.

    Args:
        base_data_dir (str, optional): Base directory for training data. Defaults to `BASE_DATA_DIR`.
        categories (list[str], optional): List of category names. Defaults to `CATEGORIES`.
        target_size (tuple[int, int], optional): Target image size for resizing. Defaults to `TARGET_IMG_SIZE`.

    Returns:
        AI.KNN_classifier.KNN: A trained KNN classifier instance. Returns `None` if
                               the training process fails (e.g., no data loaded,
                               errors during processing).
    """
    logging.info("Starting KNN model training process...")
    # Load training images and labels
    x_train_list, y_train_list = load_and_preprocess_data(base_data_dir, "train", categories, target_size)

    if not x_train_list: # Check if any training data was loaded
        logging.error("No training data loaded. Cannot train KNN model.")
        return None

    logging.info("Converting training data lists to NumPy arrays...")
    try:
        # Convert image list to a NumPy array. All images are already resized to the same shape.
        x_train = np.array(x_train_list)
        y_train = np.array(y_train_list)
    except Exception as e:
        logging.error(f"Error converting training data lists to NumPy arrays: {e}")
        return None

    # Ensure arrays are not empty after conversion
    if x_train.size == 0 or y_train.size == 0:
        logging.error("Training data or labels are empty after NumPy array conversion. Cannot train.")
        return None

    logging.info(f"Reshaping training data. Original shape of x_train (num_samples, height, width): {x_train.shape}")
    # Flatten each image into a 1D vector for KNN input.
    # Resulting shape: (num_samples, height * width)
    try:
        # Reshape from (num_samples, height, width) to (num_samples, height * width)
        x_train_flattened = x_train.reshape(x_train.shape[0], -1) # -1 infers the second dimension
        logging.info(f"Shape of flattened training data (x_train_flattened): {x_train_flattened.shape}")
    except Exception as e:
        logging.error(f"Error during training data reshaping (flattening): {e}")
        return None
    
    # Final checks before training
    if x_train_flattened.size == 0:
        logging.error("Training data is empty after flattening. Cannot train KNN model.")
        return None
    if x_train_flattened.shape[0] != y_train.shape[0]: # Ensure number of samples matches number of labels
        logging.error(f"Training data shape mismatch: X_train has {x_train_flattened.shape[0]} samples, "
                      f"but y_train has {y_train.shape[0]} labels.")
        return None

    logging.info("Training KNN model with processed data...")
    clf = KNN() # Initialize the custom KNN classifier
    clf.fit(x_train_flattened, y_train) # Train the model
    logging.info("KNN model training completed successfully.")
    return clf

def evaluate_model(clf, base_data_dir=BASE_DATA_DIR, categories=CATEGORIES, target_size=TARGET_IMG_SIZE):
    """
    Evaluates the performance of a trained KNN model on a test dataset.

    Loads and preprocesses test data similarly to training data, then makes
    predictions and calculates the accuracy.

    Args:
        clf (AI.KNN_classifier.KNN): The trained KNN classifier instance to evaluate.
        base_data_dir (str, optional): Base directory for test data. Defaults to `BASE_DATA_DIR`.
        categories (list[str], optional): List of category names. Defaults to `CATEGORIES`.
        target_size (tuple[int, int], optional): Target image size for resizing. Defaults to `TARGET_IMG_SIZE`.
    """
    if clf is None:
        logging.error("Cannot evaluate model: The provided KNN classifier is None (likely training failed).")
        return

    logging.info("Loading and preprocessing test data for model evaluation...")
    # Load test images and labels
    x_test_list, y_test_list = load_and_preprocess_data(base_data_dir, "test", categories, target_size)

    if not x_test_list: # Check if any test data was loaded
        logging.error("No testing data loaded. Cannot evaluate model.")
        return

    logging.debug("Converting test data lists to NumPy arrays...")
    try:
        x_test = np.array(x_test_list)
        y_test = np.array(y_test_list)
    except Exception as e:
        logging.error(f"Error converting test data lists to NumPy arrays: {e}")
        return

    if x_test.size == 0 or y_test.size == 0:
        logging.error("Test data or labels are empty after NumPy array conversion. Cannot evaluate.")
        return

    logging.debug(f"Reshaping test data. Original shape of x_test: {x_test.shape}")
    # Flatten test images similarly to training images
    try:
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)
        logging.debug(f"Shape of flattened test data (x_test_flattened): {x_test_flattened.shape}")
    except Exception as e:
        logging.error(f"Error during test data reshaping (flattening): {e}")
        return

    if x_test_flattened.size == 0:
        logging.error("Test data is empty after flattening. Cannot make predictions.")
        return
    if x_test_flattened.shape[0] != y_test.shape[0]: # Ensure consistency
        logging.error(f"Test data shape mismatch: X_test has {x_test_flattened.shape[0]} samples, "
                      f"but y_test has {y_test.shape[0]} labels.")
        return
    
    # Implicit check: The number of features in x_test_flattened should match what clf was trained on.
    # This is ensured by using the same target_size for resizing in both train and test data loading.

    logging.info("Making predictions on the test data...")
    predictions = clf.predict(x_test_flattened) # Get predictions from the KNN model

    # Calculate and log accuracy
    accuracy = accuracy_score(y_test, predictions)
    logging.info(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")


def main():
    """
    Main execution function when `coin_detector.py` is run as a standalone script.
    
    It orchestrates the training of a KNN model and its subsequent evaluation
    on a test set. The final accuracy is logged.
    """
    logging.info("Starting coin detection standalone script: Training and Evaluation mode.")
    
    # Train the KNN model using default parameters
    trained_clf = train_knn_model()
    
    if trained_clf:
        # If training was successful, evaluate the model
        evaluate_model(trained_clf)
    else:
        logging.error("Exiting script due to model training failure.")
        return -1 # Indicate error to the calling environment
        
    logging.info("Coin detection standalone script process finished.")
    return 0 # Indicate successful execution

if __name__ == "__main__":
    # Configure basic logging for the script.
    # This will show messages of level INFO and above, with a timestamp.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the main function and exit with its status code
    result = main()
    sys.exit(result)
