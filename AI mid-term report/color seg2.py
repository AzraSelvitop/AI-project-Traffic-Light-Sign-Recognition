import cv2
import numpy as np

# Function to detect specific colors in the image
def detect_color(image, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(image, image, mask=mask)
    return mask, result

# Function to detect circles in an image
def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(image, center, radius, (0, 255, 0), 2)  # Draw circle
    return image

# Function to detect triangles and polygons in an image
def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:  # Triangle
            cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)
        elif len(approx) > 4:  # Circle/Curve-like shapes
            cv2.drawContours(image, [approx], 0, (0, 255, 255), 2)
    return image

# HSV thresholds for multiple colors
color_ranges = {
    "red": [(np.array([0, 120, 70]), np.array([10, 255, 255])),  # Lower red range
            (np.array([170, 120, 70]), np.array([180, 255, 255]))],  # Upper red range
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "green": [(np.array([36, 100, 100]), np.array([86, 255, 255]))]
}

# Function to process an image for multiple colors and shapes
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read {image_path}")
        return
    
    # Create a copy of the original image for shape detection
    shape_image = image.copy()

    # Step 1: Color segmentation
    for color, ranges in color_ranges.items():
        print(f"Processing {color} regions...")
        combined_mask = None
        for lower, upper in ranges:
            mask, result = detect_color(image, lower, upper)
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        color_segmented = cv2.bitwise_and(image, image, mask=combined_mask)

        # Step 2: Circle and shape detection
        circle_result = detect_circles(color_segmented)
        shape_result = detect_shapes(color_segmented)

        # Display results
        cv2.imshow(f"{color.capitalize()} Mask", combined_mask)
        cv2.imshow(f"{color.capitalize()} Segmentation", color_segmented)
        cv2.imshow(f"{color.capitalize()} Circles", circle_result)
        cv2.imshow(f"{color.capitalize()} Shapes", shape_result)

    # Step 3: Display the final combined shapes
    cv2.imshow("Final Shape Detection", shape_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'C:/GTSRB/Train/resized_images/00000/00000_00008.ppm'   # Update this path for your dataset
process_image(image_path)
