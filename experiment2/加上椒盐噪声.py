import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加椒盐噪声
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    total_pixels = image.size

    # 盐噪声
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 1

    # 胡椒噪声
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_image

# 去椒盐噪声
def denoise_image(image):
    # 使用中值滤波进行去噪
    return cv2.medianBlur(image, 3)

# 读取图像
image = cv2.imread('F:/DIP/experiment2/original1.png', cv2.IMREAD_GRAYSCALE)

# 添加椒盐噪声
noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02)

# 去噪
denoised_image = denoise_image(noisy_image)

# 显示原图、噪声图和去噪图
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Denoised Image")
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.show()
