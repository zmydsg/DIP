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

# Reshape the image to be a list of pixels (data for K-Means)
Z = enhanced_image.reshape((-1, 3))
Z = np.float32(Z)

# Define criteria, number of clusters(K) and apply KMeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K = 3
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back into uint8, and make the segmented image
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape((image.shape))

# Plot histogram (frequency spectrum) of the clustered labels
plt.figure(figsize=(10, 5))
plt.hist(labels, bins=K, color='blue', alpha=0.7, rwidth=0.85)
plt.title('Frequency Spectrum of Clusters')
plt.xlabel('Cluster Index')
plt.ylabel('Frequency')
plt.show()

# Manually merge two specified clusters (e.g., clusters 0 and 1)
merged_labels = labels.copy()
merged_labels[merged_labels == 2] = 0  # Merge cluster 1 into cluster 0

# Convert merged labels back into image format
merged_centers = centers
merged_image = merged_centers[merged_labels.flatten()]
merged_image = merged_image.reshape((image.shape))

# Convert the merged image to grayscale
gray_merged = cv2.cvtColor(merged_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a mask for the foreground
_, merged_mask = cv2.threshold(gray_merged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Use the mask to extract the tiger from the original image
merged_foreground = cv2.bitwise_and(image, image, mask=merged_mask)

# Display the original image, segmented image, merged image, and the extracted foreground using OpenCV
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image (CLAHE)', enhanced_image)
cv2.imshow('Segmented Image (K-Means)', segmented_image)
cv2.imshow('Merged Image', merged_image)
cv2.imshow('Extracted Foreground (Merged)', merged_foreground)

cv2.waitKey(0)
cv2.destroyAllWindows()
