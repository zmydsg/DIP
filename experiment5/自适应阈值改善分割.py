import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'F:/DIP/experiment5/origin.png'
image = cv2.imread(image_path)

# Apply adaptive Gaussian noise removal
denosed_image = cv2.fastNlMeansDenoisingColored(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

# Apply median filtering to the denoised image
median_filtered_image = cv2.medianBlur(denosed_image, 3)

# Convert to grayscale
gray = cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to improve edge detection with varying lighting conditions
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)

# Find contours from the thresholded image
contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Sort and filter contours to keep only the largest ones to reduce noise
contours = sorted(contours, key=cv2.contourArea, reverse=True)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 400]

# Create a mask to extract the foreground
mask = np.zeros_like(gray)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Use the mask to extract the tiger from the original image
foreground = cv2.bitwise_and(median_filtered_image, median_filtered_image, mask=mask)
cv2.imwrite('F:/DIP/experiment5/adaptive-histogram_result.png', foreground)
# Display the original image, mask, and the extracted foreground
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Mask')
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Extracted Foreground')
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
