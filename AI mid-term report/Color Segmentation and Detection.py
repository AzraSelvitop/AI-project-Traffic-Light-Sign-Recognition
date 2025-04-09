import cv2
import numpy as np

def detect_color(image, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)  # Create mask for the color
    result = cv2.bitwise_and(image, image, mask=mask)  # Apply mask to image
    return mask, result

# Example for red color (two ranges for red due to HSV circular nature)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Load an image
image_path = 'C:/GTSRB/Train/resized_images/00000/00000_00000.ppm' 
image = cv2.imread(image_path)

# Detect red
mask1, result1 = detect_color(image, lower_red1, upper_red1)
mask2, result2 = detect_color(image, lower_red2, upper_red2)
combined_mask = cv2.bitwise_or(mask1, mask2)  # Combine two red masks
combined_result = cv2.bitwise_and(image, image, mask=combined_mask)

# Show results
cv2.imshow("Original", image)
cv2.imshow("Red Mask", combined_mask)
cv2.imshow("Red Segmentation", combined_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
