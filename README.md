<![CDATA[<div align="center">

# 🏗️ Dense Scaffolding Counting & Classification via CountGD

**Accurately detect, classify, and count construction scaffolding in ultra-dense, real-world site images.**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Problem Statement & Motivation](#2-problem-statement--motivation)
3.  [CountGD — The Breakthrough Solution](#3-countgd--the-breakthrough-solution)
4.  [System Architecture & Environment](#4-system-architecture--environment)
5.  [Repository Structure](#5-repository-structure)
6.  [Dataset Description](#6-dataset-description)
7.  [Hybrid Workflow: Antigravity + Kaggle](#7-hybrid-workflow-antigravity--kaggle)
8.  [Step-by-Step Guide — Kaggle Notebook Cells](#8-step-by-step-guide--kaggle-notebook-cells)
    - [Cell 1: Environment Setup](#cell-1-environment-setup--dependency-installation)
    - [Cell 2: Data Preprocessing](#cell-2-data-preprocessing--mega-dataset-consolidation)
    - [Cell 3: YOLO → ODVG Conversion](#cell-3-yolo--odvg-format-conversion)
    - [Cell 4: Training with DDP](#cell-4-distributed-training-dual-t4)
    - [Cell 5: Inference](#cell-5-inference--counting)
9.  [Configuration Reference](#9-configuration-reference)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. Introduction

This project tackles the challenging task of **detecting, classifying, and counting** three types of construction scaffolding — **IQC1524**, **L2**, and **IQC1219** — in real-world construction site photographs where scaffolding is heavily intertwined and overlapping.

The solution leverages **CountGD**, a state-of-the-art multi-modal Foundation Model built upon GroundingDINO, fine-tuned on a custom-curated dataset of **1,450 annotated images** using a memory-efficient Swin-Tiny backbone on Kaggle's dual NVIDIA Tesla T4 GPUs.

---

## 2. Problem Statement & Motivation

### The Challenge: YOLO Fails in Dense Scenes

The initial approach employed **YOLO11** (both Large and Medium variants). While YOLO excels at general object detection, it proved fundamentally inadequate for this specific domain:

| Factor | Impact |
|---|---|
| **Heavily intertwined scaffolding** | Bounding boxes overlap extensively (IoU > 0.7 between distinct objects) |
| **NMS suppression** | YOLO's Non-Maximum Suppression aggressively removes "duplicate" boxes — but in dense scaffolding, these are *valid, separate objects* |
| **Systematic undercounting** | In high-density regions (100+ scaffolds per image), YOLO missed **30–50%** of instances |

> **Core Insight:** NMS-based detectors are architecturally unsuited for counting densely packed, visually similar objects. The problem demands a *counting-first* model, not a *detection-first* model.

### Why CountGD?

CountGD reframes detection as **counting** by combining two powerful signals:

- **Text Prompts** — natural language descriptions like `"IQC1524 scaffold"`.
- **Visual Exemplars** — 2–3 reference bounding boxes per image showing the model *what* to count.

This dual-modality approach eliminates the NMS bottleneck and enables accurate counting even in extreme density (900+ objects per image).

---

## 3. CountGD — The Breakthrough Solution

CountGD is an **open-world counting** model built on top of GroundingDINO. Its key architectural innovations:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CountGD Architecture                        │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ Image Encoder │   │ Text Encoder │   │ Visual Exemplars   │  │
│  │ (Swin-Tiny)   │   │ (BERT)       │   │ (Crop Embeddings)  │  │
│  └──────┬───────┘   └──────┬───────┘   └────────┬───────────┘  │
│         │                  │                     │              │
│         │           ┌──────┴─────────────────────┘              │
│         │           │  Self-Attention Fusion                    │
│         │           │  (Text Tokens + Exemplar Tokens)          │
│         │           └──────┬──────────────────────              │
│         │                  │                                    │
│         └────────┬─────────┘                                    │
│                  │                                              │
│         ┌────────▼────────┐                                     │
│         │ Feature Enhancer │  ← Cross-Attention                 │
│         │ (Image × Text)   │                                    │
│         └────────┬────────┘                                     │
│                  │                                              │
│         ┌────────▼────────┐                                     │
│         │ Counting Decoder │  → Predicted Boxes + Count         │
│         └─────────────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Fine-Tuning Strategy (Memory-Efficient)

Given the **15 GB VRAM** limit per T4 GPU:

| Component | Status | Rationale |
|---|---|---|
| **Swin-Tiny backbone** (`backbone.0`) | 🔒 Frozen | Preserves pre-trained visual features; saves ~28M params of GPU memory |
| **BERT text encoder** (`bert`) | 🔒 Frozen | Text understanding is already robust from pre-training |
| **Projection layers** | 🔓 Trainable | Adapts feature dimensions for scaffolding domain |
| **Feature Enhancer** | 🔓 Trainable | Learns scaffolding-specific cross-modal attention |
| **Counting Decoder** | 🔓 Trainable | Learns to count scaffolding specifically |

---

## 4. System Architecture & Environment

### Platform: Hybrid Antigravity + Kaggle

```
┌───────────────────────────────┐      ┌────────────────────────────────┐
│   LOCAL (Antigravity/VSCode)  │      │      REMOTE (Kaggle)           │
│                               │      │                                │
│  • Edit .ipynb cells          │─────▶│  • 2× NVIDIA Tesla T4 (15GB)  │
│  • Write Python scripts       │      │  • 30-hour GPU quota           │
│  • Configure hyperparameters  │      │  • Dataset storage             │
│  • Git version control        │      │  • Distributed training (DDP)  │
└───────────────────────────────┘      └────────────────────────────────┘
```

### Technical Stack

| Component | Technology |
|---|---|
| Model | CountGD (GroundingDINO + Visual Exemplars) |
| Backbone | `swin_T_224_1k` (Swin-Tiny, ImageNet-1K) |
| Text Encoder | `bert-base-uncased` |
| Framework | PyTorch 2.2, CUDA, torchvision 0.17 |
| Training | Distributed Data Parallel (DDP), 2× T4 GPUs |
| Data Format | ODVG (`.jsonl`) |
| Platform | Kaggle Notebooks + Antigravity (VSCode) |

---

## 5. Repository Structure

```
CountGD/
├── config/
│   ├── cfg_scaffold_swint.py       # Training config (Swin-T, batch=2, epochs=30)
│   └── datasets_scaffold.json      # Dataset path mapping for train/val
├── datasets/                       # Dataset loaders and COCO evaluation
├── datasets_inference/             # Inference-time data transforms
├── groundingdino/                  # Core GroundingDINO module
├── models/
│   └── GroundingDINO/
│       └── ops/                    # Multi-Scale Deformable Attention (C++/CUDA)
├── tools/
│   └── yolo2odvg.py                # ★ YOLO → ODVG format converter
├── util/                           # Utilities (logging, config, misc)
├── main.py                         # Training entry point
├── engine.py                       # Train/eval loop with MAE/RMSE metrics
├── single_image_inference.py       # Single-image inference with heatmap
├── download_bert.py                # Downloads BERT weights locally
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 6. Dataset Description

### Source Data (Kaggle — YOLO Format)

The raw dataset consists of **4 sub-datasets** in YOLO format. One is excluded as irrelevant:

```
Scaffold_dataset/
├── CBS_DS_TYPE_2_IQA1900.v1i.yolov11/   ← ❌ EXCLUDED
├── IQC1524.v5i.yolov11/                 ← ✅ Class 0
├── TYPE7-L2.v5i.yolov11/                ← ✅ Class 1
└── type4-iqc1219.v3i.yolov11/           ← ✅ Class 2
```

### Consolidated Dataset (`mega_dataset`)

After preprocessing, all images are merged into a single directory with consistent class IDs:

| Class Name | Class ID | Images | Description |
|---|---|---|---|
| **IQC1524** | `0` | 312 | Standard 1524mm scaffold |
| **L2** | `1` | 424 | L2-type scaffold |
| **IQC1219** | `2` | 714 | Standard 1219mm scaffold |
| **Total** | — | **1,450** | — |

### Data Pipeline

```
YOLO .txt (per-class datasets)
        │
        ▼  [Cell 2: Merge & Remap]
  mega_dataset/  (unified YOLO format)
        │
        ▼  [Cell 3: yolo2odvg.py]
  train_scaffold.jsonl  +  valid_scaffold.jsonl  (ODVG format)
        │
        ▼  [Cell 4: Training]
  CountGD Model
```

---

## 7. Hybrid Workflow: Antigravity + Kaggle

> **Why this approach?** With only **30 hours** of Kaggle GPU time, every minute counts. Preparing code locally in Antigravity (VSCode) ensures zero wasted GPU time on debugging syntax errors or configuration issues.

### Workflow Steps

1. **Local (Antigravity/VSCode):**
   - Clone this repository.
   - Edit configuration files (`config/cfg_scaffold_swint.py`, `config/datasets_scaffold.json`).
   - Write/modify Python scripts (e.g., `tools/yolo2odvg.py`).
   - Build the `.ipynb` notebook **cell by cell** (see [Section 8](#8-step-by-step-guide--kaggle-notebook-cells)).
   - Test syntax and logic locally (CPU-only dry runs).

2. **Upload to Kaggle:**
   - Upload the completed `.ipynb` notebook.
   - Attach the Scaffold Dataset as a Kaggle Dataset input.
   - Select **GPU T4 × 2** as the accelerator.
   - Run all cells sequentially.

3. **Download Results:**
   - Retrieve `checkpoint_best_regular.pth` from the output.
   - Run inference locally or on Kaggle.

---

## 8. Step-by-Step Guide — Kaggle Notebook Cells

Create a new `.ipynb` file in Antigravity (VSCode). Add the following cells **in order**.

---

### Cell 1: Environment Setup & Dependency Installation

> Installs all required packages, compiles the C++/CUDA deformable attention operator, and downloads the BERT tokenizer.

```python
# ============================================================
# CELL 1: Environment Setup
# ============================================================
import subprocess, os, shutil

# 1a. Fresh clone CountGD repository (always get latest)
REPO_DIR = '/kaggle/working/CountGD'
os.chdir('/kaggle/working')  # Reset CWD before deleting repo
if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.run(['git', 'clone', 'https://github.com/quocthangtrann/Countthings-Project.git',
                REPO_DIR], check=True)

os.chdir(REPO_DIR)

# 1b. Verify critical fix is present (feature_map_encoder must be removed)
with open('models/GroundingDINO/groundingdino.py', 'r') as f:
    content = f.read()
assert 'feature_map_encoder' not in content, "❌ ERROR: groundingdino.py still has unused modules. Push latest code first!"
print("✅ Code verification passed.")

# 1c. Install Python dependencies
subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)

# 1d. Compile Multi-Scale Deformable Attention (C++/CUDA)
os.chdir(f'{REPO_DIR}/models/GroundingDINO/ops')
subprocess.run(['python', 'setup.py', 'build', 'install'], check=True)
os.chdir(REPO_DIR)

# 1e. Download and cache BERT tokenizer locally
subprocess.run(['python', 'download_bert.py'], check=True)

# 1f. Download pre-trained GroundingDINO Swin-T weights
WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
if not os.path.exists('groundingdino_swint_ogc.pth'):
    subprocess.run(['wget', '-q', WEIGHTS_URL], check=True)

print("✅ Environment setup complete.")
```

---

### Cell 2: Data Preprocessing — Mega-Dataset Consolidation

> Merges 3 scaffold sub-datasets into a single `mega_dataset/` directory, excludes the irrelevant `CBS_DS_TYPE_2_IQA1900` folder, and remaps all class IDs to a consistent scheme.

```python
# ============================================================
# CELL 2: Data Preprocessing — Consolidate YOLO Datasets
# ============================================================
import os, glob, shutil

print("Start preprocessing data")

search_path = '/kaggle/input/**/train/images'
all_train_dirs = glob.glob(search_path, recursive=True)

exclude_folder = "CBS DS TYPE 2 IQA1900.v1i.yolov11"
filtered_dirs = [d for d in all_train_dirs if exclude_folder not in d]

if not filtered_dirs:
    print("Error: cannot found any image")
else:
    print(f"Found {len(filtered_dirs)} dataset.\n")

    MEGA_DIR = '/kaggle/working/mega_dataset'
    for split in ['train', 'valid', 'test']:
        os.makedirs(f"{MEGA_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{MEGA_DIR}/{split}/labels", exist_ok=True)

    total_img_counter = 0
    class_names = ['IQC1524', 'L2', 'IQC1219']

    for ds_idx, train_img_dir in enumerate(filtered_dirs):
        ds_root = os.path.dirname(os.path.dirname(train_img_dir))
        folder_name = os.path.basename(ds_root).lower()

        if 'iqc1524' in folder_name:
            class_id = 0
        elif 'l2' in folder_name:
            class_id = 1
        elif 'iqc1219' in folder_name:
            class_id = 2
        else:
            continue

        print(f"Preprocessing Dataset: {class_names[class_id]} (ID: {class_id}) ...")
        folder_img_count = 0

        for split in ['train', 'valid', 'test']:
            src_imgs = f"{ds_root}/{split}/images"
            src_lbls = f"{ds_root}/{split}/labels"
            if not os.path.exists(src_imgs): continue

            for img_name in os.listdir(src_imgs):
                if not img_name.endswith(('.jpg', '.jpeg', '.png')): continue

                base_name = os.path.splitext(img_name)[0]
                lbl_name = f"{base_name}.txt"

                src_img_path = os.path.join(src_imgs, img_name)
                src_lbl_path = os.path.join(src_lbls, lbl_name)

                if os.path.exists(src_lbl_path):
                    prefix = f"ds{ds_idx}_"
                    dst_img = f"{MEGA_DIR}/{split}/images/{prefix}{img_name}"
                    dst_lbl = f"{MEGA_DIR}/{split}/labels/{prefix}{lbl_name}"

                    shutil.copy2(src_img_path, dst_img)

                    with open(src_lbl_path, 'r') as f_in, open(dst_lbl, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if len(parts) > 1:
                                parts[0] = str(class_id)
                                f_out.write(" ".join(parts) + "\n")

                    folder_img_count += 1
                    total_img_counter += 1

        print(f"   -> Merge successful: {folder_img_count} images.\n")

    print(f"✅ Finished. Total {total_img_counter} images in {MEGA_DIR}.")
```

---

### Cell 3: YOLO → ODVG Format Conversion

> CountGD requires ODVG (`.jsonl`) annotations instead of YOLO `.txt` files. This cell converts bounding boxes from normalized YOLO format to absolute pixel coordinates, assigns text captions, and randomly selects 2–3 visual exemplar boxes per image.

```python
# ============================================================
# CELL 3: Convert YOLO Annotations → ODVG JSONL
# ============================================================
import os, json, random
from PIL import Image

def convert_yolo_to_odvg(img_dir, lbl_dir, output_file):
    """Convert YOLO .txt labels to CountGD ODVG .jsonl format.
    
    Produces the exact format expected by ODVGDataset (VG mode):
      - "filename": relative image name (joined with root in config)
      - "grounding": {"regions": [{"bbox": [x1,y1,x2,y2], "phrase": "..."}]}
      - "exemplars": [[x1,y1,x2,y2], ...]
    """
    class_map = {
        0: "IQC1524 scaffold",
        1: "L2 scaffold",
        2: "IQC1219 scaffold"
    }

    jsonl_data = []
    print(f"Processing {img_dir}...")

    for lbl_name in os.listdir(lbl_dir):
        if not lbl_name.endswith('.txt'): continue

        img_name = lbl_name.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)

        if not os.path.exists(img_path): continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        # Collect all boxes with their class phrases
        regions = []
        all_boxes = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue

                cls_id = int(parts[0])
                if cls_id not in class_map: continue

                cx, cy, nw, nh = map(float, parts[1:5])
                x1 = (cx - nw / 2) * img_w
                y1 = (cy - nh / 2) * img_h
                x2 = (cx + nw / 2) * img_w
                y2 = (cy + nh / 2) * img_h

                bbox = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                phrase = class_map[cls_id]
                regions.append({"bbox": bbox, "phrase": phrase})
                all_boxes.append(bbox)

        if not regions:
            continue

        # Select 2-3 random exemplar boxes
        num_exemplars = min(3, len(all_boxes))
        exemplars = random.sample(all_boxes, num_exemplars)

        record = {
            "filename": img_name,           # Relative path (root is in config)
            "grounding": {"regions": regions},
            "exemplars": exemplars
        }
        jsonl_data.append(record)

    with open(output_file, 'w') as f:
        for record in jsonl_data:
            f.write(json.dumps(record) + '\n')
    print(f"✅ Saved {len(jsonl_data)} records → {output_file}")

# --- Execute conversion ---
MEGA = '/kaggle/working/mega_dataset'

convert_yolo_to_odvg(
    img_dir=f'{MEGA}/train/images',
    lbl_dir=f'{MEGA}/train/labels',
    output_file='/kaggle/working/train_scaffold.jsonl'
)

convert_yolo_to_odvg(
    img_dir=f'{MEGA}/valid/images',
    lbl_dir=f'{MEGA}/valid/labels',
    output_file='/kaggle/working/valid_scaffold.jsonl'
)
```

---

### Cell 4: Distributed Training (Dual T4)

> Launches fine-tuning with PyTorch DDP across both T4 GPUs. The backbone and BERT encoder are frozen — only the projection layers, Feature Enhancer, and Counting Decoder are trained.

```python
# ============================================================
# CELL 4: Distributed Training (DDP on 2× T4 GPUs)
# ============================================================
import subprocess, os, json

os.chdir('/kaggle/working/CountGD')

# 4a. Ensure datasets_scaffold.json has all required fields
config_path = 'config/datasets_scaffold.json'
cfg = {
    "train": [
        {
            "name": "scaffold_train",
            "dataset_mode": "odvg",
            "root": "/kaggle/working/mega_dataset/train/images",
            "anno": "/kaggle/working/train_scaffold.jsonl",
            "label_map": None
        }
    ],
    "val": [
        {
            "name": "scaffold_val",
            "dataset_mode": "odvg",
            "root": "/kaggle/working/mega_dataset/valid/images",
            "anno": "/kaggle/working/valid_scaffold.jsonl",
            "label_map": None
        }
    ]
}
with open(config_path, 'w') as f:
    json.dump(cfg, f, indent=4)
print("✅ Dataset config verified.")

# 4b. Launch training with DDP
cmd = [
    'python', '-m', 'torch.distributed.launch',
    '--nproc_per_node=2',
    '--use_env',
    'main.py',
    '--config_file', 'config/cfg_scaffold_swint.py',
    '--datasets', config_path,
    '--output_dir', '/kaggle/working/output',
    '--pretrain_model_path', 'groundingdino_swint_ogc.pth',
    '--find_unused_params',
]

print("🚀 Starting distributed training on 2× T4 GPUs...")
print("Command:", ' '.join(cmd))

process = subprocess.run(cmd)

if process.returncode == 0:
    print("✅ Training completed successfully!")
    print("Best checkpoint: /kaggle/working/output/checkpoint_best_regular.pth")
else:
    print(f"❌ Training failed with return code {process.returncode}")
```

**Key Training Hyperparameters** (from `config/cfg_scaffold_swint.py`):

| Parameter | Value | Notes |
|---|---|---|
| `backbone` | `swin_T_224_1k` | Swin-Tiny, memory-efficient |
| `batch_size` | `2` | Per-GPU; prevents OOM on T4 |
| `epochs` | `30` | Fits within 30-hour limit |
| `lr` | `1e-4` | Base learning rate |
| `freeze_keywords` | `['backbone.0', 'bert']` | Freezes image + text encoders |
| `num_queries` | `900` | Max detectable objects per image |
| `box_threshold` | `0.23` | Confidence threshold |
| `lr_drop` | `10` | LR decay step |

---

### Cell 5: Inference & Counting

> Runs the trained model on a test image and generates a density heatmap overlay with the predicted count.

```python
# ============================================================
# CELL 5: Single Image Inference
# ============================================================
import subprocess, os, glob

os.chdir('/kaggle/working/CountGD')

# Auto-select the first test image (or replace with a specific path)
test_images = glob.glob('/kaggle/working/mega_dataset/test/images/*.jpg')
if not test_images:
    test_images = glob.glob('/kaggle/working/mega_dataset/valid/images/*.jpg')
TEST_IMAGE = test_images[0]
print(f"Testing with: {TEST_IMAGE}")

cmd = [
    'python', 'single_image_inference.py',
    '--config', 'config/cfg_scaffold_swint.py',
    '--pretrain_model_path', '/kaggle/working/output/checkpoint_best_regular.pth',
    '--image_path', TEST_IMAGE,
    '--text', 'IQC1524 scaffold',
    '--confidence_thresh', '0.23',
    '--output_image_name', '/kaggle/working/result_heatmap.jpg',
]

print("🔍 Running inference...")
subprocess.run(cmd, check=True)

# Display the result
from IPython.display import display, Image as IPImage
display(IPImage(filename='/kaggle/working/result_heatmap.jpg'))
```

> **Tip:** To count a different scaffold type, change the `--text` argument to `"L2 scaffold"` or `"IQC1219 scaffold"`.

---

## 9. Configuration Reference

### `config/datasets_scaffold.json`

Maps dataset names to the ODVG annotation files generated in Cell 3:

```json
{
    "train": [
        {
            "name": "scaffold_train",
            "dataset_mode": "odvg",
            "root": "/kaggle/working/mega_dataset/train/images",
            "anno": "/kaggle/working/train_scaffold.jsonl",
            "label_map": null
        }
    ],
    "val": [
        {
            "name": "scaffold_val",
            "dataset_mode": "odvg",
            "root": "/kaggle/working/mega_dataset/valid/images",
            "anno": "/kaggle/working/valid_scaffold.jsonl",
            "label_map": null
        }
    ]
}
```

### `config/cfg_scaffold_swint.py` — Key Sections

```python
# --- Backbone (Frozen) ---
backbone = "swin_T_224_1k"
freeze_keywords = ['backbone.0', 'bert']

# --- Training ---
batch_size = 2
epochs = 30
lr = 0.0001
lr_drop = 10
weight_decay = 0.0001

# --- Model Architecture ---
num_queries = 900          # Supports ultra-dense images
enc_layers = 6
dec_layers = 6
hidden_dim = 256

# --- Inference Thresholds ---
box_threshold = 0.23
text_threshold = 0
```

---

## 10. Acknowledgements

- **[CountGD](https://arxiv.org/abs/2407.04619)** — Ammar, Sindhu, & Lukasiewicz. *"CountGD: Multi-Modal Open-World Counting"* (2024).
- **[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)** — IDEA Research. The base detection architecture.
- **[Kaggle](https://kaggle.com)** — Free dual-T4 GPU compute for training.
- **[Antigravity](https://marketplace.visualstudio.com/items?itemName=Google.antigravity)** — AI-powered coding assistant for local development.

---

<div align="center">

**Built with ❤️ for the construction industry.**

*If this project helps your research, please ⭐ the repo!*

</div>
]]>
