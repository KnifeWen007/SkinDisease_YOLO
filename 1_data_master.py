import os
import cv2
import pandas as pd
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# === 配置路径 ===
RAW_IMG = "raw_data/images"
RAW_MASK_T1 = "raw_data/masks_t1"
RAW_MASK_T2 = "raw_data/masks_t2"
CSV_PATH = "raw_data/labels.csv"

# 输出路径
OUT_YOLO = "datasets/yolo_seg"
OUT_CLS = "datasets/cls_cropped"

# 类别映射
# YOLO 类别 (0是病灶，1-4是特征)
YOLO_CLASSES = {
    "lesion": 0, "streaks": 1, "globules": 2,
    "pigment_network": 3, "milia_like_cyst": 4
}
# 分类类别 (根据 CSV)
CLS_CLASSES = ["nevus", "melanoma", "seborrheic_keratosis"]


def get_yolo_polygon(mask, w, h):
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            pts = []
            for p in cnt:
                x, y = p[0][0] / w, p[0][1] / h
                pts.append(f"{min(max(x, 0), 1):.6f} {min(max(y, 0), 1):.6f}")
            if len(pts) > 4: polys.append(" ".join(pts))
    return polys


def crop_lesion(img, mask_path):
    """A+C核心：利用Mask裁剪病灶，背景涂黑"""
    if not os.path.exists(mask_path): return img  # 没Mask就返回原图
    mask = cv2.imread(mask_path, 0)
    # 背景变黑
    masked = cv2.bitwise_and(img, img, mask=mask)
    # 裁剪 ROI
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return masked
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # 稍微外扩 10 像素
    pad = 10
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = min(img.shape[1] - x, w + 2 * pad), min(img.shape[0] - y, h + 2 * pad)
    return masked[y:y + h, x:x + w]


def main():
    # 1. 读取 CSV
    df = pd.read_csv(CSV_PATH)
    # 构造 image_id 到 label 的映射
    # 逻辑：melanoma=1 -> "melanoma", seborrheic=1 -> "seborrheic", 否则 "nevus"
    id_to_label = {}
    for idx, row in df.iterrows():
        img_id = row['image_id']
        if row['melanoma'] == 1.0:
            label = "melanoma"
        elif row['seborrheic_keratosis'] == 1.0:
            label = "seborrheic_keratosis"
        else:
            label = "nevus"
        id_to_label[img_id] = label

    # 2. 准备目录
    for split in ['train', 'val']:
        # YOLO 目录
        os.makedirs(f"{OUT_YOLO}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_YOLO}/labels/{split}", exist_ok=True)
        # CLS 目录
        for cls_name in CLS_CLASSES:
            os.makedirs(f"{OUT_CLS}/{split}/{cls_name}", exist_ok=True)

    # 3. 划分数据集
    all_imgs = [f for f in os.listdir(RAW_IMG) if f.endswith('.jpg')]
    train_imgs, val_imgs = train_test_split(all_imgs, test_size=0.2, random_state=42)

    print("开始由 Task 1 分割引导的数据清洗...")

    for split, img_list in zip(['train', 'val'], [train_imgs, val_imgs]):
        for img_name in tqdm(img_list, desc=split):
            basename = os.path.splitext(img_name)[0]
            img_path = os.path.join(RAW_IMG, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]

            # --- A. 准备 YOLO 数据 (Task 1 + 2) ---
            txt_lines = []

            # T1: 整体病灶
            t1_path = os.path.join(RAW_MASK_T1, f"{basename}_segmentation.png")
            if os.path.exists(t1_path):
                mask = cv2.imread(t1_path, 0)
                for poly in get_yolo_polygon(mask, w, h):
                    txt_lines.append(f"0 {poly}")

            # T2: 细微特征
            for feat_name, fid in YOLO_CLASSES.items():
                if fid == 0: continue
                # 注意：ISIC文件名可能是 basename_feature.png
                f_path = os.path.join(RAW_MASK_T2, f"{basename}_{feat_name}.png")
                if os.path.exists(f_path):
                    mask = cv2.imread(f_path, 0)
                    for poly in get_yolo_polygon(mask, w, h):
                        txt_lines.append(f"{fid} {poly}")

            # 保存 YOLO 数据
            if txt_lines:  # 只有有标注才拿来训练YOLO
                shutil.copy(img_path, f"{OUT_YOLO}/images/{split}/{img_name}")
                with open(f"{OUT_YOLO}/labels/{split}/{basename}.txt", 'w') as f:
                    f.write("\n".join(txt_lines))

            # --- B. 准备分类数据 (Task 3 - A+C模式) ---
            # 只有当 CSV 里有这个图，且 T1 分割存在时，才做 A+C 裁剪
            if basename in id_to_label and os.path.exists(t1_path):
                label = id_to_label[basename]
                # 裁剪出纯净病灶 (A+C 的关键步骤!)
                cropped_img = crop_lesion(img, t1_path)
                # 保存到对应分类文件夹
                save_path = f"{OUT_CLS}/{split}/{label}/{img_name}"
                cv2.imwrite(save_path, cropped_img)


if __name__ == "__main__":
    main()