import os
import json
import random
from PIL import Image

def convert_yolo_to_odvg(img_dir, lbl_dir, output_file):
    class_map = {
        0: "IQC1524 scaffold",
        1: "L2 scaffold",
        2: "IQC1219 scaffold"
    }

    jsonl_data = []
    print(f"start processing {img_dir}...")

    for lbl_name in os.listdir(lbl_dir):
        if not lbl_name.endswith('.txt'): continue
        
        img_name = lbl_name.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
        if not os.path.exists(img_path): continue
        
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        
        class_boxes = {}
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
                
                cls_name = class_map[cls_id]
                if cls_name not in class_boxes:
                    class_boxes[cls_name] = []
                class_boxes[cls_name].append([x1, y1, x2, y2])
        
        # Tạo bản ghi cho CountGD (Bốc ngẫu nhiên 3 hộp làm mẫu)
        for cls_name, boxes in class_boxes.items():
            num_exemplars = min(3, len(boxes))
            exemplars = random.sample(boxes, num_exemplars)
            
            record = {
                "file_name": img_path,
                "caption": f"{cls_name} .", # Cú pháp bắt buộc của mô hình
                "grounding_dict": {f"{cls_name}": boxes},
                "exemplars": exemplars
            }
            jsonl_data.append(record)

    with open(output_file, 'w') as f:
        for record in jsonl_data:
            f.write(json.dumps(record) + '\n')
    print(f"save {len(jsonl_data)} at {output_file}")

if __name__ == "__main__":
    # Đường dẫn trỏ tới dữ liệu TRÊN KAGGLE
    convert_yolo_to_odvg(
        img_dir='/kaggle/input/mega_dataset/train/images',
        lbl_dir='/kaggle/input/mega_dataset/train/labels',
        output_file='/kaggle/working/train_scaffold.jsonl'
    )
    # Nếu bạn có tập Valid, làm tương tự:
    convert_yolo_to_odvg(
        img_dir='/kaggle/input/mega_dataset/valid/images',
        lbl_dir='/kaggle/input/mega_dataset/valid/labels',
        output_file='/kaggle/working/valid_scaffold.jsonl'
    )