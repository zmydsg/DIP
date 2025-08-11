
# Digital Image Processing Project (数字图像处理项目)

This repository contains the coursework for a Digital Image Processing course, including experiments, final papers, and reports.
本仓库包含数字图像处理课程的作业，包括实验、最终论文和报告。

## Directory Structure (目录结构)

*   `DIP_finalpaper/`: Contains the final research paper on contour detection and image segmentation. (包含关于轮廓检测和图像分割的最终研究论文。)
*   `experiment1/`: RGB and HSV color space-based image segmentation. (基于RGB和HSV颜色空间的图像分割。)
*   `experiment2/`: Noise addition (Gaussian, salt-and-pepper) and histogram equalization. (添加噪声（高斯、椒盐）和直方图均衡化。)
*   `experiment3/`: Camera calibration and distortion correction. (相机标定和畸变校正。)
*   `experiment4/`: Edge detection with Canny and adaptive histogram equalization. (使用Canny进行边缘检测和自适应直方图均衡化。)
*   `experiment5/`: Various image segmentation techniques, including K-means, Mean-Shift, and superpixel segmentation. (各种图像分割技术，包括K-means、Mean-Shift和超像素分割。)
*   `experiment6/`: Image segmentation using brightness and multi-scale min-cut. (使用亮度和多尺度最小割进行图像分割。)
*   `张朋洋2022104334数图报告+论文/`: Contains all the experimental reports and the final paper. (包含所有实验报告和最终论文。)

## Final Paper: Contour Detection and Hierarchical Image Segmentation (最终论文：轮廓检测与层次化图像分割)

This section details the final project, which focuses on implementing and evaluating the "Tiny Pb" contour detection algorithm and comparing it against classical edge detection methods.
本部分详细介绍了最终项目，其重点是实现和评估 "Tiny Pb" 轮廓检测算法，并将其与经典的边缘检测方法进行比较。

### Project Goal (项目目标)

The goal is to explore modern contour detection techniques and understand their performance relative to traditional algorithms like Canny and Sobel. The project implements the lightweight "Tiny Pb" detector and uses it as a basis for hierarchical image segmentation.
目标是探索现代轮廓检测技术，并了解其相对于像Canny和Sobel这样的传统算法的性能。该项目实现了轻量级的 "Tiny Pb" 检测器，并将其用作层次化图像分割的基础。

### Directory Contents (目录内容)

* `Contour_Detection_and_Hierarchical_Image_Segmentation.pdf`: The final research paper in English. (最终的英文研究论文。)
* `DIP_FINAL_PAPER_张朋洋.pdf`: The final research paper in Chinese. (最终的中文研究论文。)
* `flowchart.png`: A flowchart illustrating the algorithm's workflow. (展示算法工作流程的流程图。)
* `Tiny-Pb-Contour-Detection/`: Contains the core implementation of the project. (包含项目的核心实现。)
  * `Tiny-Pb.py`: The Python script for the "Tiny Pb" contour detector. ( "Tiny Pb" 轮廓检测器的Python脚本。)
  * `BSDS500/`: The Berkeley Segmentation Data Set 500, used for training and testing the model. (伯克利分割数据集500，用于训练和测试模型。)
  * `Results/`: Directory for storing the output images from the detector. (用于存储检测器输出图像的目录。)
* `Processor/`: Contains implementations of baseline algorithms for comparison. (包含用于比较的基线算法的实现。)
  * `Canny边缘检测.py`: An implementation of the Canny edge detector. (Canny边缘检测器的实现。)
  * `Sobelbaselinegenerator.py`: A script to generate baseline results using the Sobel operator. (使用Sobel算子生成基线结果的脚本。)

## Dependencies (依赖)

This project primarily uses the following Python libraries:
本项目主要使用以下Python库：

-   OpenCV (`cv2`)
-   NumPy
-   Matplotlib

You can install them using pip:
您可以使用pip安装它们：

```bash
pip install opencv-python numpy matplotlib
```

## Usage (使用方法)

Navigate to the respective experiment's directory and run the Python scripts.
导航到相应实验的目录并运行Python脚本。

```bash
cd experiment1
python RGB阈值分割.py
```
