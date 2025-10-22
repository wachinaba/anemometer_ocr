from ultralytics import YOLO
import os

# --- Configuration ---
# Load the trained model.
# This script assumes 'best.pt' is in the same directory.
MODEL_PATH = 'model.pt'

# Set the path to the image you want to analyze.
# Modify this to your image file.
IMAGE_PATH = '7seg_test02.jpg' 

# --- Main Execution ---
def main():
    """
    Loads the YOLO model, runs prediction on the specified image,
    and prints the detected objects.
    """
    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    # Check if the image file exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at '{IMAGE_PATH}'")
        print("Please update the 'IMAGE_PATH' variable in this script.")
        return

    # Load the model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    print(f"Model '{MODEL_PATH}' loaded successfully.")
    print(f"Running inference on '{IMAGE_PATH}'...")

    # Run prediction
    try:
        results = model.predict(source=IMAGE_PATH, save=True)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # --- Process and Display Results ---
    print("-" * 30)
    print("Inference Complete.")
    
    # The 'results' object contains detailed information.
    # For object detection, it's a list with one element.
    result = results[0]

    # Print the path where the output image with bounding boxes is saved
    print(f"Output image saved in: {result.save_dir}")

    # Print detected objects and their confidence
    if len(result.boxes) == 0:
        print("No objects detected.")
    else:
        print("Detected objects:")
        # Sort boxes by their x-coordinate to read left-to-right
        sorted_boxes = sorted(result.boxes, key=lambda box: box.xyxy[0][0])
        
        detected_string = ""
        for box in sorted_boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            detected_string += class_name
            print(f"  - Class: {class_name}, Confidence: {confidence:.2f}")
        
        print(f"\nDetected string (left to right): {detected_string}")

    print("-" * 30)


if __name__ == "__main__":
    main()
