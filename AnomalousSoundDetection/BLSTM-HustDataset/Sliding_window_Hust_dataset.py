import os
import numpy as np
import scipy.io
from sklearn.utils import resample

# --- ตั้งค่า Seed สำหรับควบคุมการสุ่ม ---
np.random.seed(42)

# --- ตั้งค่า Path และตัวแปร ---
dataset_path = r"/home/bt/Desktop/Jin/HUST bearing a practical dataset for ball bearing fault diagnosis/HUST bearing dataset"
save_path    = r"/home/bt/Desktop/Jin/Processed_Data"
target_samples = 64983
window_length  = 2048
stride         = 80

# เตรียม Dictionary สำหรับเก็บข้อมูลแต่ละคลาส
data_dict = {'Normal': [], 'InnerRace': [], 'OuterRace': [], 'Ball': []}

# --- 1. โหลดข้อมูล ---
print("กำลังเริ่มโหลดไฟล์ .mat และตัด Window...")

mat_files = [f for f in os.listdir(dataset_path) if f.endswith(".mat")]
print(f"พบไฟล์ .mat ทั้งหมด {len(mat_files)} ไฟล์\n")

for filename in mat_files:
    # คัดแยกคลาสจากชื่อไฟล์
    if   filename.startswith('N'):                          label = 'Normal'
    elif filename.startswith('I') and not filename.startswith('IB') and not filename.startswith('IO'): label = 'InnerRace'
    elif filename.startswith('O') and not filename.startswith('OB'):                                   label = 'OuterRace'
    elif filename.startswith('B'):                          label = 'Ball'
    elif filename.startswith('IB') or filename.startswith('IO'): label = 'InnerRace'  # compound faults → InnerRace
    elif filename.startswith('OB'):                         label = 'OuterRace'        # compound faults → OuterRace
    else:
        print(f"  [ข้าม] ไม่รู้จักไฟล์: {filename}")
        continue

    file_path = os.path.join(dataset_path, filename)
    try:
        mat_data = scipy.io.loadmat(file_path)
    except Exception as e:
        print(f"  [ข้าม] โหลดไฟล์ {filename} ไม่ได้: {e}")
        continue

    # หาชื่อตัวแปรในไฟล์ .mat
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if not keys:
        print(f"  [ข้าม] ไม่พบตัวแปรใน {filename}")
        continue

    signal = mat_data[keys[0]].flatten()

    # ตัด Window
    windows_added = 0
    for i in range(0, len(signal) - window_length + 1, stride):
        data_dict[label].append(signal[i:i + window_length])
        windows_added += 1

    print(f"  {filename:15s} → คลาส {label:10s} | signal={len(signal):>8,} | windows={windows_added:>5,}")

# --- 2. สรุปผลก่อนสุ่ม ---
print("\n--- รายงานจำนวนข้อมูลก่อนสุ่ม (Class Balancing) ---")
min_samples = None
for cls, data in data_dict.items():
    n = len(data)
    print(f"คลาส {cls:10s} มีจำนวน {n:>7,} ชิ้น")
    if n > 0:
        min_samples = n if min_samples is None else min(min_samples, n)

if min_samples is None or min_samples == 0:
    print("\n[ERROR] ไม่มีข้อมูลในบางคลาส กรุณาตรวจสอบ dataset_path")
    exit(1)

# ปรับ target ถ้า target_samples มากกว่าข้อมูลที่มี
actual_target = min(target_samples, min_samples)
if actual_target < target_samples:
    print(f"\n[WARNING] target_samples={target_samples:,} มากเกินไป → ปรับเป็น {actual_target:,} (จากคลาสที่มีน้อยสุด)")

# --- 3. ทำการสุ่มให้เท่ากัน ---
balanced_data = []
labels        = []
classes_to_balance = ['Normal', 'InnerRace', 'OuterRace', 'Ball']

print(f"\n--- กำลังสุ่มข้อมูลให้เหลือคลาสละ {actual_target:,} ชิ้น ---")
for idx, cls in enumerate(classes_to_balance):
    arr     = np.array(data_dict[cls])
    sampled = resample(arr, n_samples=actual_target, replace=False, random_state=42)
    balanced_data.append(sampled)
    labels.extend([idx] * actual_target)
    print(f"คลาส {cls:10s} → คงเหลือ {len(sampled):,} ชิ้น")

# --- 4. บันทึกไฟล์ ---
final_data   = np.concatenate(balanced_data, axis=0)
final_labels = np.array(labels)

os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, "balanced_data.npy"),   final_data)
np.save(os.path.join(save_path, "balanced_labels.npy"), final_labels)

print("\n============================================")
print(f"บันทึกไฟล์สำเร็จที่: {save_path}")
print(f"จำนวนข้อมูลรวมทั้งหมดหลังสุ่มคือ: {len(final_data):,} ชิ้น")
print(f"Shape: {final_data.shape}  |  Labels shape: {final_labels.shape}")
print("============================================")