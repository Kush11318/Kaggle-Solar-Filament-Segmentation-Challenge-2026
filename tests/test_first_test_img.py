import sys, os, cv2
sys.path.append(r"d:\Kaggle")
from scripts.run_unleashed_multiroute_consensus import MAGFiLOUnleashedEngine
from src.paths import get_dataset_paths

root_ckpt = r"d:\Kaggle\checkpoints"
yolo_m = os.path.join(root_ckpt, "yolo11m_staple_best.pt")
yolo_x = os.path.join(root_ckpt, "yolo11x_seg_best.pt")
b4 = os.path.join(root_ckpt, "crop_unet_effnet_b4_512_mined_best.pt")
r34 = os.path.join(root_ckpt, "crop_unet_resnet34_512_mined_best.pt")

engine = MAGFiLOUnleashedEngine(yolo_m, yolo_x, b4, r34, device="cpu")

_, _, test_img_dir = get_dataset_paths()
test_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
sample_fn = test_files[0]
sample_path = os.path.join(test_img_dir, sample_fn)
print(f"Testing on {sample_fn}...", flush=True)
img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
insts = engine.process_solar_image(img, conf_m=0.30, conf_x=0.30, enable_route2_faint=True, enable_route5_spine=True)
print(f"Successfully segmented {len(insts)} instances!", flush=True)
for i, inst in enumerate(insts[:5], 1):
    print(f"  Inst {i}: Area = {inst['area']}, Score = {inst['score']:.3f}", flush=True)
print("SUCCESS!")

