import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image

# 设置后端
matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入自定义模块
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])  # 预定义颜色
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 尝试平滑轮廓
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_masks(image, masks, scores, box_coords=None, borders=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca(), borders=borders)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 模型参数
    checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    # 构建模型
    sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # 图像和掩码的路径
    image_folder = 'results/swinir_classical_sr_x8/'
    mask_folder = 'results/water_masks_8/'
    save_path = 'results/seg-8-sr/'
    os.makedirs(save_path, exist_ok=True)

    # 遍历所有图像文件
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(('.png', '.jpg', '.jpeg')):
            # 构建完整的图像路径
            image_path = os.path.join(image_folder, image_filename)
            print(f"Processing image: {image_path}")

            # 使用cv2读取图像
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
            # 调整图像尺寸为(256, 256)
            image = cv2.resize(image, (256, 256))
            print("Resized image shape:", image.shape)

            # 构建对应的掩码路径
            mask_path = os.path.join(mask_folder, f"{os.path.splitext(image_filename)[0]}_only_water_mask.png")
            print(f"Using mask: {mask_path}")

            # 使用cv2读取掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
            # 确保掩膜的尺寸与输入图像一致
            mask = cv2.resize(mask, (256, 256))  # 调整为 (256, 256)

            # 确保掩码是二值的
            mask_binary = np.where(mask > 0, 1, 0).astype(np.uint8)

            # 获取掩码的坐标
            input_mask = mask_binary

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image)
                masks, scores, _ = predictor.predict(point_coords=None, point_labels=None,
                                                     mask_input=input_mask[None, :, :],
                                                     multimask_output=False)


            print(f"masks shape: {masks.shape}")

            # 显示分割结果
            #show_masks(image, masks[0], scores)

            # 应用形态学开运算去噪
            mask_image = (masks[0] * 255).astype(np.uint8)  # 转换为 0-255

            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)  # 创建一个5x5的结构元素
            mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)

            # Resize the mask image to 500x500
            mask_image = cv2.resize(mask_image, (500, 500))

            # 保存分割后的掩码
            save_mask_path = os.path.join(save_path, f"{os.path.splitext(image_filename)[0]}.png")
            cv2.imwrite(save_mask_path, mask_image)  # 使用cv2保存图像

        print("Processing complete.")
