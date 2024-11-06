# data.py

import os
import numpy as np
import cv2

def data_generator(image_dir, mask_dir, batch_size, image_size):
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)
    
    while True:
        for i in range(0, len(image_files), batch_size):
            batch_images = []
            batch_masks = []
            
            for j in range(i, min(i + batch_size, len(image_files))):
                image = cv2.imread(os.path.join(image_dir, image_files[j]))
                mask = cv2.imread(os.path.join(mask_dir, mask_files[j]), cv2.IMREAD_GRAYSCALE)
                
                image = cv2.resize(image, image_size)
                mask = cv2.resize(mask, image_size)
                
                image = image / 255.0  # 归一化
                mask = mask / 255.0    # 掩码也归一化

                batch_images.append(image)
                batch_masks.append(mask[:, :, np.newaxis])  # 保持掩码尺寸为 [H, W, 1]
            
            yield np.array(batch_images), np.array(batch_masks)
