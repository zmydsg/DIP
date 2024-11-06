# train_unet.py

import os
from tensorflow.keras.models import load_model
from model import unet_model
from data import data_generator
import cv2

# 定义路径
image_dir = 'F:/DIP/Unettrain/origial images/'
mask_dir = 'F:/DIP/Unettrain/mask/'
image_size = (256, 256)  # 定义图像的大小，确保所有图像和掩码尺寸相同
batch_size = 8
epochs = 50

# 创建数据生成器
train_gen = data_generator(image_dir, mask_dir, batch_size, image_size)
steps_per_epoch = len(os.listdir(image_dir)) // batch_size

# 初始化U-Net模型
model = unet_model(input_size=(256, 256, 3))

# 训练模型
model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs)

# 保存训练好的模型
model.save('unet_yellow_segmentation.h5')
