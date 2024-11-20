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

# Manually annotate regions for labeling using OpenCV drawing functions
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
        labels_points.append((x, y))

# Create a copy for annotation
annotated_image = enhanced_image.copy()
labels_points = []

cv2.namedWindow('Manual Annotation')
cv2.setMouseCallback('Manual Annotation', draw_circle)

while True:
    cv2.imshow('Manual Annotation', annotated_image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()

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

# Convert the segmented image to grayscale
gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a mask for the foreground
_, kmeans_mask = cv2.threshold(gray_segmented, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Use the mask to extract the tiger from the original image
foreground = cv2.bitwise_and(image, image, mask=kmeans_mask)

# Display the original image, segmented image, and the extracted foreground using OpenCV
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image (CLAHE)', enhanced_image)
cv2.imshow('Segmented Image (K-Means)', segmented_image)
cv2.imshow('Extracted Foreground', foreground)

cv2.waitKey(0)
cv2.destroyAllWindows()
