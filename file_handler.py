"""
File Handler Utility.

This module provides basic file handling operations.
Currently, it contains a function to append data to a specified file.
This was initially used by `data_transformation.py` for saving pixel data,
but that functionality has since been integrated directly into `data_transformation.py`.
This module is retained in case its file operation utilities are needed elsewhere.
"""

def file(filename, data):
    """
    Appends the given data string to the specified file.

    If the file does not exist, it will be created. If it exists, data
    will be added to the end of the file.

    Args:
        filename (str): The path to the file where data will be appended.
        data (str): The string data to append to the file.

    Returns:
        None.
    
    Raises:
        IOError: If any error occurs during file operations (e.g., permission issues).
        TypeError: If `filename` or `data` are not strings.
    """
    # The 'a+' mode opens the file for reading and appending.
    # The file is created if it does not exist. The stream is positioned at the end of the file.
    try:
        with open(filename, 'a+') as f:
            f.write(data)
    except TypeError as e:
        # Log or handle the error appropriately if logging was configured system-wide
        # For now, re-raise to make the caller aware of the type issue.
        # print(f"Error: filename and data must be strings. Details: {e}") # Avoid print in library code
        raise TypeError(f"Invalid type for filename or data. Must be strings. Original error: {e}")
    except IOError as e:
        # print(f"Error appending to file {filename}: {e}") # Avoid print in library code
        raise IOError(f"Failed to open or write to file '{filename}'. Original error: {e}")
    except Exception as e: # Catch any other unexpected errors
        # print(f"An unexpected error occurred with file {filename}: {e}")
        raise Exception(f"An unexpected error occurred with file '{filename}'. Original error: {e}")

# Example usage (typically would not be in a library file like this):
# if __name__ == '__main__':
#     try:
#         file("test_log.txt", "This is a test entry.\n")
#         file("test_log.txt", "Another test entry.\n")
#         print("Successfully wrote to test_log.txt")
#         # Example of error handling
#         # file(123, "This will fail") 
#     except Exception as e:
#         print(f"An error occurred during example usage: {e}")
