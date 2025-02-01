import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import matplotlib
# 设置后端为TkAgg
matplotlib.use('TkAgg')

# 导入SAM2相关模块
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# 定义显示掩码的函数
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


# 定义显示点的函数
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


# 定义显示框的函数
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# 定义显示掩码的函数
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


# 主程序入口
if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())

    checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # 设置输入图像和掩膜的目录
    input_image_dir = 'results/SR_x2/'
    input_mask_dir = "results/water_masks"
    output_mask_dir = "results/seg_X2"

    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    # 获取输入目录中所有图像文件的列表
    image_files = [f for f in os.listdir(input_image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    # 遍历每个图像文件
    for image_file in image_files:
        # 构造完整文件路径
        image_path = os.path.join(input_image_dir, image_file)

        # 使用 OpenCV 加载图像
        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        print("image shape:", image.shape)

        print(f"Loading image from: {image_path}")

        # 检查图像是否成功加载
        if image is None:
            print(f"Error loading image: {image_path}. Please check the file path or file format.")
            continue  # 跳过此文件，处理下一个文件

        # 调整图像尺寸为(256, 256)
        image = cv2.resize(image, (256, 256))
        print("Resized image shape:", image.shape)

        # 获取输入掩膜文件的列表
        mask_files = [f for f in os.listdir(input_mask_dir) if f.endswith('.png') or f.endswith('.jpg')]
        for mask_file in mask_files:
            # 构造完整掩膜路径
            mask_path = os.path.join(input_mask_dir, mask_file)

            # 加载掩膜图像
            input_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式加载掩膜图像
            # 确保掩膜的尺寸与输入图像一致
            input_mask = cv2.resize(input_mask, (256, 256))  # 调整为 (256, 256)
            # 确保掩膜的尺寸与输入图像一致
            # input_mask = cv2.resize(input_mask, (image.shape[1], image.shape[0]))  # 调整为 (W, H)

            # 二值化掩膜，确保值为0或255
            #_, input_mask = cv2.threshold(input_mask, 127, 255, cv2.THRESH_BINARY)  # 采用127作为阈值

            print("Binary mask shape:", input_mask.shape)  # 应该是 (H, W)
            #print("Unique values in binary mask:", np.unique(input_mask))  # 应该只有0和255

            # 将掩膜转换为适合输入模型的形式
            #input_mask = np.expand_dims(input_mask, axis=0) / 255.0  # 变为 (1，H, W)
            input_mask = np.expand_dims(input_mask, axis=0)/ 255.0  # 变为 (1, H, W, 1)

            print("Normalized input mask shape:", input_mask.shape)



            # 使用加载的掩膜进行预测
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image)
                masks, scores, _ = predictor.predict(point_coords=None, point_labels=None,
                                                 mask_input=input_mask,
                                                 multimask_output=False)
                print("Masks shape:", masks.shape)




            # 显示分割结果
            show_masks(image, masks, scores)

            # 保存分割掩码图
            output_mask_file = os.path.join(output_mask_dir, os.path.splitext(image_file)[0] + '_seg.png')
            cv2.imwrite(output_mask_file, (masks[0] * 255).astype(np.uint8))