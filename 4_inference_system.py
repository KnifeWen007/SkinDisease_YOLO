import cv2
import torch
import numpy as np
import os
import random
import glob
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image

# === 配置部分 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 路径配置 (请确认路径无误)
YOLO_PATH = 'runs/segment/yolo_task1_23/weights/best.pt'
CLS_MODEL_PATH = 'cls_task3_model.pth'

# 2. 标签定义
CLS_MAP = {
    'melanoma': 'Melanoma (恶性黑色素瘤)',
    'nevus': 'Nevus (普通痣)',
    'seborrheic_keratosis': 'SK (脂溢性角化)'
}
CLS_INDEX_NAMES = ['melanoma', 'nevus', 'seborrheic_keratosis']

# === 设备检测 ===
print("=== 设备检测 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"当前设备: {torch.cuda.current_device()} - {torch.cuda.get_device_name()}")
    print(f"设备类型: GPU")
else:
    print("设备类型: CPU")

print(f"运行设备: {device}")

# === 模型文件大小检查 ===
print("\n=== 模型文件大小检查 ===")
def get_file_size(file_path):
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.2f} MB"
        else:
            return f"{size_bytes/(1024**3):.2f} GB"
    return "文件不存在"

print(f"YOLO 模型文件: {YOLO_PATH}")
print(f"文件大小: {get_file_size(YOLO_PATH)}")

print(f"分类模型文件: {CLS_MODEL_PATH}")
print(f"文件大小: {get_file_size(CLS_MODEL_PATH)}")

# === 初始化模型 ===
print("\n=== 初始化模型 ===")
print(f"正在加载 YOLO 模型: {YOLO_PATH} ...")
if not os.path.exists(YOLO_PATH): raise FileNotFoundError(f"找不到 YOLO: {YOLO_PATH}")

# 强制YOLO模型使用指定设备
yolo_model = YOLO(YOLO_PATH)
if torch.cuda.is_available():
    yolo_model.to('cuda')
    print("YOLO模型已加载到GPU")
else:
    print("YOLO模型运行在CPU上")

print(f"正在加载分类模型: {CLS_MODEL_PATH} ...")
if not os.path.exists(CLS_MODEL_PATH): raise FileNotFoundError("找不到分类模型")

cls_model = models.mobilenet_v3_large()
cls_model.classifier[3] = torch.nn.Linear(cls_model.classifier[3].in_features, 3)
cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=device))
cls_model.to(device).eval()

# 强制触发GPU内存分配（如果使用GPU）
if torch.cuda.is_available():
    # 创建一个虚拟输入来触发GPU内存分配
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        _ = cls_model(dummy_input)
    print("分类模型已加载到GPU并触发内存分配")
else:
    print("分类模型运行在CPU上")

print("模型加载完毕！")

# === 显存使用情况 ===
print("\n=== 显存使用情况 ===")
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated() / 1024**3
    reserved_memory = torch.cuda.memory_reserved() / 1024**3
    max_allocated_memory = torch.cuda.max_memory_allocated() / 1024**3
    free_memory = total_memory - allocated_memory
    
    print(f"总显存: {total_memory:.2f} GB")
    print(f"当前显存使用: {allocated_memory:.2f} GB")
    print(f"峰值显存使用: {max_allocated_memory:.2f} GB")
    print(f"已缓存显存: {reserved_memory:.2f} GB")
    print(f"剩余显存: {free_memory:.2f} GB")
    print(f"显存使用率: {allocated_memory/total_memory*100:.1f}%")
    
    # 检查模型是否真的在GPU上
    print("\n=== 模型设备检查 ===")
    yolo_on_gpu = any(param.device.type == 'cuda' for param in yolo_model.model.parameters())
    cls_on_gpu = any(param.device.type == 'cuda' for param in cls_model.parameters())
    
    print(f"YOLO模型在GPU上: {yolo_on_gpu}")
    print(f"分类模型在GPU上: {cls_on_gpu}")
    
    if not (yolo_on_gpu and cls_on_gpu):
        print("警告: 模型可能没有正确加载到GPU！")
else:
    print("当前运行在CPU上，无显存信息")

print(f"模型加载完成，准备进行推理...")

def diagnose(img_path, true_label_hint=None):
    """
    img_path: 图片路径
    true_label_hint: 真实标签提示 (用于验证)
    """
    if not os.path.exists(img_path): return

    # 1. 读取原图
    img_cv = cv2.imread(img_path)
    if img_cv is None: return

    # 复制一份纯净的原图用于展示（左侧）
    original_display = img_cv.copy()

    # 2. YOLO 分割
    results = yolo_model(img_cv, verbose=False)[0]
    # 获取带有掩膜和框的处理图（右侧）
    annotated_img = results.plot()

    lesion_mask = None
    if results.masks is not None:
        for i, cls_id in enumerate(results.boxes.cls):
            if int(cls_id) == 0:
                lesion_mask = results.masks.data[i].cpu().numpy()
                break
        if lesion_mask is None and len(results.masks.data) > 0:
            lesion_mask = results.masks.data[0].cpu().numpy()

    diagnosis_text = "No Lesion"

    # 3. 如果找到了病灶 -> 分类
    if lesion_mask is not None:
        lesion_mask = cv2.resize(lesion_mask, (img_cv.shape[1], img_cv.shape[0]))
        lesion_mask = (lesion_mask > 0.5).astype(np.uint8) * 255
        masked = cv2.bitwise_and(img_cv, img_cv, mask=lesion_mask)
        x, y, w, h = cv2.boundingRect(lesion_mask)

        if w > 0 and h > 0:
            roi = masked[y:y + h, x:x + w]
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = transform(roi_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = cls_model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)
                pred_label_key = CLS_INDEX_NAMES[pred_idx]
                diagnosis_text = f"{CLS_MAP[pred_label_key]} {conf.item():.1%}"

                # 控制台打印
                print(f"\n[{os.path.basename(img_path)}] 诊断详情:")
                print(f"  真实标签: {true_label_hint}")
                print(f"  预测结果: {diagnosis_text}")
                print(f"  详细概率: {[f'{n}:{p:.2f}' for n, p in zip(CLS_INDEX_NAMES, probs)]}")

            # 在右侧图上画出裁剪框
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- 4. 可视化合成 (Side-by-Side) ---

    # 给图片加个小标题
    cv2.putText(original_display, "Original Image", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(annotated_img, "AI Segmentation", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # 左右拼接
    # axis=1 表示横向拼接
    combined_body = np.hstack((original_display, annotated_img))

    # 顶部信息栏
    header_h = 120
    header = np.zeros((header_h, combined_body.shape[1], 3), dtype=np.uint8)

    # 颜色逻辑：恶性=红，其他=绿
    color = (0, 0, 255) if "Melanoma" in diagnosis_text else (0, 255, 0)

    # 写入诊断结果
    cv2.putText(header, f"Diagnosis: {diagnosis_text}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    # 写入真实标签（如果是测试模式）
    if true_label_hint:
        cv2.putText(header, f"(Reference Ground Truth: {true_label_hint})", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    # 上下拼接
    final_display = np.vstack((header, combined_body))

    # --- 智能缩放 ---
    # 因为两张图拼在一起很宽，所以我们主要限制宽度或者高度
    max_w = 1600  # 最大宽度
    max_h = 850  # 最大高度

    h, w = final_display.shape[:2]
    scale = 1.0

    # 如果太宽或太高，计算缩放比例
    if w > max_w or h > max_h:
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h)  # 取较小的缩放比例，保证能完全装下

        new_w = int(w * scale)
        new_h = int(h * scale)
        final_display = cv2.resize(final_display, (new_w, new_h))

    cv2.imshow("Skin AI System - Comparison View", final_display)
    print("\n按任意键继续 (按 'q' 退出)...")
    return cv2.waitKey(0)


if __name__ == "__main__":
    # 定向测试模式
    cls_root = "datasets/cls_cropped/val"
    yolo_root = "datasets/yolo_seg/images/val"
    categories = ['melanoma', 'nevus', 'seborrheic_keratosis']

    print(">>> 启动对比视图测试 (Comparison View) <<<")

    while True:
        for cat in categories:
            cat_dir = os.path.join(cls_root, cat)
            if not os.path.exists(cat_dir): continue
            files = os.listdir(cat_dir)
            if not files: continue

            target_file = random.choice(files)
            full_img_path = os.path.join(yolo_root, target_file)

            if not os.path.exists(full_img_path):
                full_img_path = full_img_path.replace("val", "train")

            if os.path.exists(full_img_path):
                key = diagnose(full_img_path, true_label_hint=cat)
                if key == ord('q'):
                    exit()