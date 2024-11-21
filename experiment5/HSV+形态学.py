import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'F:/DIP/experiment5/origin.png'
image = cv2.imread(image_path)

# Apply Adaptive Histogram Equalization (CLAHE) for better contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
lab[:, :, 0] = clahe.apply(lab[:, :, 0])
enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Convert the image to RGB
rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

# Convert to HSV color space to help isolate the tiger
hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

# Define the range for orange color to segment the tiger
lower_orange = np.array([0, 80, 80])
upper_orange = np.array([30, 255, 255])

# Define the range for white color to segment the tiger
lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 60, 255])

# Define the range for black color to segment the tiger's stripes and tail
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# Create masks using the HSV range for orange, white, and black
mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
mask_black = cv2.inRange(hsv_image, lower_black, upper_black)

# Combine the masks to include orange, white, and black parts of the tiger
combined_mask = cv2.bitwise_or(mask_orange, mask_white)
combined_mask = cv2.bitwise_or(combined_mask, mask_black)

# Apply morphological operations to improve the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

# Use the mask to extract the tiger from the original image
foreground = cv2.bitwise_and(image, image, mask=combined_mask)

# Display the original image and the extracted foreground
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Extracted Tiger Foreground')
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
