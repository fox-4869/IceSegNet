# IceSegNet：面向遥感图像河冰语义分割的阶段感知动态卷积核网络

<div align="center">

[![论文](https://img.shields.io/badge/论文-Applied%20Soft%20Computing-blue?style=for-the-badge&logo=elsevier)](https://www.sciencedirect.com/science/article/pii/S1568494625014334)
[![代码](https://img.shields.io/badge/代码-GitHub-black?style=for-the-badge&logo=github)](https://github.com/fox-4869/IceSegNet)
[![框架](https://img.shields.io/badge/框架-MMSegmentation-orange?style=for-the-badge)](https://github.com/open-mmlab/mmsegmentation)
[![许可证](https://img.shields.io/badge/许可证-Apache%202.0-green?style=for-the-badge)](LICENSE)

[English](README.md) | **中文**

</div>

> **IceSegNet: A stage-aware dynamic kernel network for river ice segmentation in remote sensing imagery**
>
> 邬开俊、周鼎举\*、杜娟娟、武月莲、张立东
>
> *Applied Soft Computing, Vol. 186, 2026* | [📄 论文链接](https://www.sciencedirect.com/science/article/pii/S1568494625014334)

---

## 🏔️ 核心亮点

- 🎯 **阶段感知卷积核更新模块** — 三个结构差异化的更新阶段（细节保留 → 过渡稳定 → 语义纯化），前馈网络宽度逐阶递减（2048 → 1024 → 512），FFN 参数量减少 **41.7%**，同时 mIoU 提升 **0.96%**。
- 🧠 **UPerSCA-MTL 解码头** — 在 UPerNet 基础上引入轻量级空间交叉注意力（SCA）和多任务学习（MTL），联合优化语义分割与边缘检测任务。
- 🏆 **最优性能** — 在 NWPU_YRCC2 数据集达 **93.81% mIoU**（超越 K-Net 基线 1.19%，超越 Mask2Former 1.31%）；在 NWPU_YRCC_EX 数据集达 **93.56% mIoU**（超越 K-Net 0.43%）。

---

## 📋 摘要

对遥感图像中的河冰进行精确分割，对于量化冰盖覆盖率、支撑冰凌灾害早期预警与风险评估具有重要意义。然而，大尺度变化、冰水光谱相似性以及边界模糊等问题带来了显著挑战。

IceSegNet 提出两项核心创新：

1. **阶段感知卷积核更新模块**：通过三个功能各异的阶段逐步精炼特征，并采用逐阶递减的前馈网络宽度，在减少参数量的同时提升分割精度。
2. **UPerSCA-MTL 解码头**：融合空间交叉注意力与边缘检测多任务学习，显著提升边界分割精度。

在两个黄河冰凌数据集（NWPU_YRCC2 与 NWPU_YRCC_EX）上与 18 种前沿方法对比，IceSegNet 均取得最优结果。

---

## 🏗️ 网络结构

![网络结构图](IceSegNet_model.png)

### 阶段感知卷积核更新模块

| 阶段 | 功能定位 | FFN 隐藏层宽度 |
|------|---------|--------------|
| I | **细节保留** — 保存细粒度纹理与边缘线索 | 2048 |
| II | **过渡稳定** — 衔接低层细节与高层语义 | 1024 |
| III | **语义纯化** — 紧凑类别判别嵌入，滤除冗余 | 512 |

### UPerSCA-MTL 解码头

- **PPM（金字塔池化模块）**：多尺度上下文聚合
- **SCA（空间交叉注意力）**：水平 + 垂直全局池化，捕获轴向方向依赖，增强冰水边界区分能力
- **深度可分离卷积**：在保留特征质量的前提下降低计算复杂度
- **多任务输出头**：语义分割分支 + 边缘检测分支并行输出；边缘监督信号由 GT mask 梯度自动生成，**无需额外人工标注**

---

## 📊 主要实验结果

### NWPU_YRCC2 数据集（4类：陆地 / 水体 / 岸冰 / 流冰）

| 方法 | 发表会议 | mIoU (%) | PA (%) | mFscore (%) | FPS | 参数量 (M) |
|------|---------|----------|--------|-------------|-----|-----------|
| U-Net | MICCAI 2015 | 61.62 | 80.59 | 75.55 | 3.05 | 29 |
| PSPNet | CVPR 2017 | 90.70 | 94.28 | 95.07 | 9.41 | 47 |
| DeepLabv3+ | ECCV 2018 | 90.37 | 94.74 | 94.92 | 5.42 | 60 |
| SegFormer | NeurIPS 2021 | 86.99 | 92.61 | 92.98 | 4.18 | 82 |
| Mask2Former | CVPR 2022 | 92.50 | 96.12 | 96.09 | 3.75 | 216 |
| DINOv2+Rein | CVPR 2025 | 89.86 | 94.01 | 94.63 | 2.21 | 317 |
| K-Net *（基线）* | NeurIPS 2021 | 92.62 | 95.78 | 96.15 | 3.58 | 245 |
| **IceSegNet *（本文）*** | ASOC 2025 | **93.81** | **96.31** | **96.79** | 3.43 | 247 |

### NWPU_YRCC_EX 数据集（3类：冰体 / 水体 / 其他）

| 方法 | mIoU (%) | PA (%) | mFscore (%) |
|------|----------|--------|-------------|
| PSPNet | 85.42 | 92.54 | 92.13 |
| DeepLabv3+ | 88.04 | 93.93 | 93.63 |
| Mask2Former | 93.26 | 96.56 | 96.50 |
| K-Net *（基线）* | 93.13 | 96.49 | 96.43 |
| **IceSegNet *（本文）*** | **93.56** | **96.73** | **96.67** |

---

## ⚙️ 环境安装

### 依赖要求

- Python ≥ 3.8
- PyTorch ≥ 1.12（含 CUDA）
- MMEngine、MMCV ≥ 2.0、MMSegmentation

### 逐步安装

**第一步：创建 conda 环境**

```bash
conda create -n icesegnet python=3.8 -y
conda activate icesegnet
```

**第二步：安装 PyTorch**（以 CUDA 11.6 为例）

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

**第三步：安装 MMEngine 和 MMCV**

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**第四步：安装 MMSegmentation**

```bash
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
cd ..
```

**第五步：克隆本仓库并安装依赖**

```bash
git clone https://github.com/fox-4869/IceSegNet.git
cd IceSegNet
pip install -r requirements.txt
```

---

## 📁 数据准备

下载数据集：

- **NWPU_YRCC2**：[https://github.com/nwpulab113/NWPUYRCC2](https://github.com/nwpulab113/NWPUYRCC2)
- **NWPU_YRCC_EX**：[https://github.com/nwpulab113/NWPUYRCCEX](https://github.com/nwpulab113/NWPUYRCCEX)

按如下目录结构组织数据：

```
data/
├── NWPU_YRCC2_JPG1/
│   ├── train/             # 训练图像（.jpg）
│   ├── train_labels/      # 训练标注（.png）
│   ├── val/               # 验证图像
│   └── val_labels/        # 验证标注
└── NWPU_YRCC_EX/
    ├── train/
    ├── train_labels/
    ├── val/
    └── val_labels/
```

### 注册自定义模块

**1. 复制数据集定义文件**

```bash
cp datasets/NWPU_YRCC2_JPG1.py mmsegmentation/mmseg/datasets/
cp datasets/NWPU_YRCC.py       mmsegmentation/mmseg/datasets/
```

在 `mmseg/datasets/__init__.py` 中添加导入：

```python
from .NWPU_YRCC2_JPG1 import NWPU_YRCC2_JPG1
from .NWPU_YRCC import NWPU_YRCC
```

**2. 复制模型组件**

```bash
cp models/sefpn.py              mmsegmentation/mmseg/models/necks/
cp models/uper_att_plus_head.py mmsegmentation/mmseg/models/decode_heads/
```

分别在 `mmseg/models/necks/__init__.py` 和 `mmseg/models/decode_heads/__init__.py` 中完成注册。

---

## 🚀 训练

**单卡训练**

```bash
python tools/train.py configs/icesegnet-config.py
```

**多卡训练**（论文使用 2× RTX 3090）

```bash
bash tools/dist_train.sh configs/icesegnet-config.py 2
```

关键训练参数：

| 参数 | 设置 |
|------|------|
| 优化器 | AdamW（β₁=0.9，β₂=0.999） |
| 初始学习率 | 6×10⁻⁵ |
| 权重衰减 | 5×10⁻⁴ |
| 批量大小 | 4/GPU × 2 GPU = 8 |
| 最大迭代次数 | 60,000 |
| 学习率策略 | 线性预热（500 iter）+ 余弦退火 |
| 裁剪尺寸 | 512 × 512 |
| 主干网络初始化 | ImageNet-22K 预训练 Swin-L |

---

## 🧪 测试与评估

**标准评估**

```bash
python tools/test.py configs/icesegnet-config.py /path/to/checkpoint.pth
```

**测试时增强（多尺度 + 翻转）**

```bash
python tools/test.py configs/icesegnet-config.py /path/to/checkpoint.pth --tta
```

评估指标：**mIoU**、**mDice**、**mFscore**、**PA**、**BFscore**

---

## 📂 仓库结构

```
IceSegNet/
├── configs/
│   └── icesegnet-config.py        # 完整训练与评估配置
├── datasets/
│   ├── NWPU_YRCC2_JPG1.py         # 4类数据集（陆地/水体/岸冰/流冰）
│   └── NWPU_YRCC.py               # 3类数据集（其他/水体/岸冰）
├── models/
│   ├── sefpn.py                   # SEFPN 颈部（含BN+ReLU归一化）
│   └── uper_att_plus_head.py      # UPerSCA-MTL 解码头
├── tools/                         # 训练与测试脚本（来自 MMSeg）
├── README.md                      # 英文 README
└── README_CN.md                   # 中文 README
```

---

## 📝 配置文件说明

模型以 MMSegmentation 的 `EncoderDecoder` 框架组织：

```
主干网络：  SwinTransformer-L   (embed_dims=192, depths=[2,2,18,2])
颈部：     SEFPN               (in_channels=[192,384,768,1536], out_channels=768)
解码头：   IterativeDecodeHead
             ├── kernel_generate_head: UPerATTPLUSHead
             └── kernel_update_head:  3× KernelUpdateHead
                   阶段 I：   feedforward_channels=2048，num_heads=16
                   阶段 II：  feedforward_channels=1024，num_heads=8
                   阶段 III： feedforward_channels=512， num_heads=4
辅助头：   FCNHead             (in_channels=768, channels=256)
```

---

## 📖 引用

如果本工作对您的研究有所帮助，请引用：

```bibtex
@article{wu2026icesegnet,
  title     = {IceSegNet: A stage-aware dynamic kernel network for river ice
               segmentation in remote sensing imagery},
  author    = {Wu, Kaijun and Zhou, Dingju and Du, Juanjuan and
               Wu, Yuelian and Zhang, Lidong},
  journal   = {Applied Soft Computing},
  volume    = {186},
  pages     = {114120},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.asoc.2025.114120}
}
```

---

## 🙏 致谢

本研究得到以下项目支持：甘肃省自然科学基金重点项目（23JRRA860）、内蒙古自治区重点研发和成果转化计划项目（2023YFSH0043、2023YFDZ0043、2023YFDZ0054）、兰州交通大学重点科研项目（ZDYF2304）、甘肃省优秀研究生"创新之星"项目（2025CXZX-682）。

本代码基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 构建，感谢 [K-Net](https://github.com/ZwwWayne/K-Net) 提供的动态卷积核框架基础。

---

## 📬 联系方式

**通讯作者**：周鼎举 — [dingjuzhou@163.com](mailto:dingjuzhou@163.com)

兰州交通大学，兰州 730070
