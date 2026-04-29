# Dense Scaffolding Counting & Classification via CountGD

## 1. Introduction & Objective

**Objective:** Accurately detect, classify, and count 3 types of construction scaffolding: `IQC1524`, `L2`, and `IQC1219` in real-world construction site images.

This project utilizes CountGD, a multi-modal Foundation Model based on GroundingDINO, adapted specifically for the complex and high-density environment of construction sites.

## 2. Problem Statement & Breakthrough Solution

### The Challenge
Initially, YOLO11 (Large and Medium versions) was used to tackle this problem. However, the construction site environment proved to be extremely complex with heavily intertwined scaffolding. This caused YOLO's Non-Maximum Suppression (NMS) mechanism to unintentionally suppress valid bounding boxes, leading to severe undercounting in high-density areas.

### The Solution
We transitioned to **CountGD** to overcome YOLO's limitations. CountGD provides several key advantages for this specific use case:
- **Multi-Modal Foundation:** Combines Text Prompts with Visual Exemplars (providing sample reference boxes for the model to "look at" and count).
- **Adaptive Cropping:** Supports advanced techniques for counting in ultra-dense images by automatically dividing the image into patches, preventing OOM errors and maintaining high recall for intertwined objects.

## 3. System Architecture & Environment

### Platform & Hardware
- **Platform:** Kaggle Notebook
- **Hardware:** Dual GPU NVIDIA Tesla T4 (2 x 15GB VRAM)

### Technical Stack
- **Model:** CountGD (Swin-T backbone)
- **Framework:** PyTorch, CUDA
- **Data Format:** ODVG (`.jsonl`)

---

## 4. Usage Instructions

### Step 1: Installation and Compilation

1. **Clone the repository and install dependencies:**
   ```bash
   git clone <repository-url>
   cd CountGD
   pip install -r requirements.txt
   ```

2. **Compile the C++/CUDA core:**
   Navigate to the operations directory and compile the Multi-Scale Deformable Attention algorithm:
   ```bash
   cd models/GroundingDINO/ops/
   python setup.py build install
   cd ../../../
   ```

3. **Download Pre-trained Weights:**
   Download the Swin-T version of the pre-trained weights (`groundingdino_swint_ogc.pth`) and place it in the project root.

### Step 2: Data Preparation

CountGD requires the ODVG (`.jsonl`) format instead of the standard YOLO format. 

Use the provided custom script to convert your YOLO annotations (`.txt`) into the required ODVG format:
```bash
python tools/yolo2odvg.py
```
*Note: This script recalculates absolute bounding box coordinates, assigns required text captions (e.g., `"IQC1524 scaffold ."`), and randomly extracts 2-3 scaffolding bounding boxes per image to serve as "Visual Exemplars" for the model.*

### Step 3: Configuration Tuning

1. **Dataset Mapping:**
   Ensure the data mapping file `config/datasets_scaffold.json` correctly points to your training and validation `.jsonl` files.

2. **Training Configuration (`config/cfg_scaffold_swint.py`):**
   The configuration is tailored for Dual T4 GPUs:
   - **Backbone:** `swin_T_224_1k` (Lightweight version)
   - **Batch Size:** `2` (Lowered to prevent Out Of Memory errors)
   - **Epochs:** `30`
   - **Learning Rate:** `1e-4`
   - **Freezing:** `freeze_keywords = ['backbone.0', 'bert']` (Freezes the visual encoder and text encoder to only train the Counting head).

### Step 4: Training

Utilize Distributed Data Parallel (DDP) to leverage the power of dual GPUs. Run the following command as a background job:

```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
  --config_file config/cfg_scaffold_swint.py \
  --datasets config/datasets_scaffold.json
```

### Step 5: Inference

Test the model on real-world images using the best saved checkpoint (`checkpoint_best_regular.pth`). 

Input the text prompt and exemplar bounding boxes. Ensure the **Adaptive Cropping** feature is enabled so the model automatically crops high-density images into smaller patches, counts the objects, and aggregates the final results (crucial for images with 900+ scaffolding instances).

```bash
python single_image_inference.py \
  --config_file config/cfg_scaffold_swint.py \
  --checkpoint_path checkpoint_best_regular.pth \
  # ... Add other required inference arguments as needed
```
