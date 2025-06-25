import os
from collections import Counter

dataset_path = "dataset"  # เปลี่ยนเป็น path โฟลเดอร์ของคุณ

class_counts = Counter()

for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(image_files)

# แสดงผล
for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{class_name:35} : {count} ภาพ")
