from PIL import Image
import csv

# 指定图片路径与导出CSV路径
image_path = 'G:/DIP_final/MeanShift_py/MeanShift_py/original3.png'  # 输入图片路径
csv_path = 'data.csv'    # 导出的csv文件路径

# 打开图片
with Image.open(image_path) as img:
    
    
    width, height = img.size
    
    # 打开CSV文件，以写入模式
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 若只想导出坐标，不需要颜色，可使用：
        writer.writerow(['x', 'y'])
        
        # 如果需要导出像素点的颜色值（RGB），可以写入表头
        # writer.writerow(['x', 'y', 'R', 'G', 'B'])
        
        # 遍历每个像素点
        for y in range(height):
            for x in range(width):
                # 获取像素点的RGB值
                # r, g, b = img.getpixel((x, y))
                
                # 写入csv行数据，如果不需要颜色可以仅写入[x, y]
                writer.writerow([x, y])
