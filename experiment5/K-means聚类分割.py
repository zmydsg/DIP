import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'F:/DIP/experiment5/origin.png'
image = cv2.imread(image_path)

# # Apply adaptive Gaussian noise removal
# image = cv2.fastNlMeansDenoisingColored(image0, None, h=6, templateWindowSize=5, searchWindowSize=21)

# Reshape the image to be a list of pixels (data for K-Means)
Z = image.reshape((-1, 3))
Z = np.float32(Z)

# Define criteria, number of clusters(K) and apply KMeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.6)
K = 3
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back into uint8, and make the segmented image
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape((image.shape))

# Convert the segmented image to grayscale
gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a mask for the foreground
_, kmeans_mask = cv2.threshold(gray_segmented, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Use the mask to extract the tiger from the original image
foreground = cv2.bitwise_and(image, image, mask=kmeans_mask)
cv2.imwrite('F:/DIP/experiment5/K-means_result.png', foreground)
# Display the original image, segmented image, and the extracted foreground
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Segmented Image (K-Means)')
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Extracted Foreground')
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
