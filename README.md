# 🏥 基于分割引导级联网络的皮肤病变智能诊断系统
### (Skin Lesion Intelligent Diagnosis System based on Segmentation-Guided Cascade Network)

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📖 项目简介 (Introduction)

本项目针对 **ISIC 2017 皮肤病变分析挑战赛** 任务，提出了一种**“分割引导分类 (Segmentation-Guided Classification)”** 的级联深度学习框架。

传统的皮肤病分类模型容易受到背景噪声（如体毛、标记尺、正常皮肤纹理）的干扰（即“尺子效应”）。本项目通过以下流程解决该问题：
1.  **实例分割 (Task 1 & 2)**: 使用 **YOLOv8** 模型精准定位病灶区域（Lesion）及特定的皮肤镜特征（如条纹、色素网）。
2.  **ROI 提取**: 基于分割掩码，通过自动化脚本裁剪出纯净的**感兴趣区域 (ROI)**，去除背景干扰。
3.  **细粒度分类 (Task 3)**: 将裁剪后的纯净图像输入 **MobileNet/EfficientNet** 进行良恶性（黑色素瘤/脂溢性角化病/黑色素痣）分类。

---

## 🛠️ 环境安装 (Installation)

本项目建议使用 Conda 管理环境。

### 1. 创建虚拟环境
```bash
conda create -n skin_ai python=3.9 -y
conda activate skin_ai
```

### 2. 安装 PyTorch
推荐安装 CUDA 版本以支持 GPU 加速（以下为 CUDA 11.8 示例）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装依赖库
```bash
pip install ultralytics opencv-python matplotlib pandas tqdm scikit-learn
```

---

## 📂 项目结构 (Project Structure)

```text
SkinDisease_YOLO/
├── datasets/                 # [自动生成] 数据处理后的输出目录
│   ├── cls_cropped/          # (Task 3) 裁剪后的纯净病灶分类数据集
│   └── yolo_seg/             # (Task 1&2) YOLO 格式的分割数据集
├── raw_data/                 # [必须手动准备] 原始数据目录
│   ├── images/               # ISIC 原始 JPG 图像
│   ├── masks_t1/             # 第一类掩码 (病灶整体分割)
│   ├── masks_t2/             # 第二类掩码 (特征属性分割)
│   └── labels.csv            # ISIC 2017 分类标签文件
├── runs/                     # 训练过程日志与权重保存目录
├── 1_data_master.py          # [Step 1] 数据清洗、格式转换与 ROI 裁剪脚本
├── 2_train_yolo.py           # [Step 2] YOLOv8 分割模型训练脚本
├── 3_train_cls.py            # [Step 3] CNN 分类模型训练脚本
├── 4_inference_system.py     # [Step 4] 全流程级联推理系统
├── main.py                   # 项目主入口 (可选)
├── skin_multitask.yaml       # YOLO 数据集配置文件
└── README.md                 # 项目说明文档
```

---

## 💾 数据准备 (Data Preparation)

本项目依赖 **ISIC 2017 Challenge** 数据集。请严格按照以下步骤准备数据，否则代码无法运行。

1.  **访问数据源**: [ISIC 2017 Challenge Dataset](https://challenge.isic-archive.com/data/#2017)
2.  **下载并解压**至项目根目录下的 `raw_data` 文件夹：

    | 官方名称 | 存放位置 | 说明 |
    | :--- | :--- | :--- |
    | **Training Data** | `raw_data/images/` | 包含约 2000 张 `.jpg` 图片 |
    | **Training Ground Truth (Part 1)** | `raw_data/masks_t1/` | 包含 `_segmentation.png` 掩码 |
    | **Training Ground Truth (Part 2)** | `raw_data/masks_t2/` | 包含特征属性掩码 |
    | **Training Ground Truth (Part 3)** | `raw_data/labels.csv` | **重命名**原始 CSV 文件为此名 |

---

## 🚀 运行指南 (Quick Start)

### 第一步：数据工程 (Data Engineering)
处理原始数据，生成 YOLO 训练数据并裁剪出分类用的 ROI 图像。
```bash
python 1_data_master.py
```
> **输出**: 
> *   `datasets/yolo_seg`: 包含 YOLO 所需的 images 和 txt labels。
> *   `datasets/cls_cropped`: 包含按类别（Melanoma, Nevus, SK）分类的裁剪后图像。

### 第二步：训练分割网络 (Train Segmentation)
训练 YOLOv8 模型以提取病灶掩码。
```bash
python 2_train_yolo.py
```
> **输出**: 模型权重保存在 `runs/segment/yolo_task1_2/weights/best.pt`。

### 第三步：训练分类网络 (Train Classification)
训练分类网络（MobileNet/EfficientNet）以识别良恶性。
```bash
python 3_train_cls.py
```
> **输出**: 模型权重保存在 `cls_task3_model.pth`，并输出验证集准确率。

### 第四步：系统推理 (Inference System)
加载上述两个训练好的模型，对图片进行端到端的诊断。
```bash
python 4_inference_system.py
```
> **功能**: 读取测试图片 -> YOLO 分割病灶 -> 裁剪 ROI -> CNN 分类 -> 输出诊断报告。

---

## ⚙️ 配置文件 (Configuration)

请确保根目录下包含 `skin_multitask.yaml` 用于 YOLO 训练配置：

```yaml
path: ./datasets/yolo_seg
train: images/train
val: images/val

names:
  0: lesion           # 病灶本体
  1: streaks          # 条纹特征
  2: globules         # 球状物
  3: pigment_network  # 色素网
  4: milia_like_cyst  # 粟粒样囊肿
```

---

## 📝 引用与致谢 (Acknowledgments)

*   数据来源: **International Skin Imaging Collaboration (ISIC) Archive**
*   YOLO 实现: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---
*Created by [KnifeWen]*
```