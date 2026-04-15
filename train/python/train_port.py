from ultralytics import YOLO
import torch
from pathlib import Path


# ========= 你主要改这里 =========
MODEL_FILE = "yolov8n.pt"
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "yolo_port" / "dataset.yaml"
PROJECT_DIR = ROOT_DIR / "runs"
RUN_NAME = "detect_retrain"

EPOCHS = 100
IMGSZ = 640
BATCH = 8

# 设备设置：
# "cpu"  -> 强制用 CPU
# 0      -> 强制用第 0 张 CUDA 显卡（通常是 NVIDIA 独显）
# None   -> 让 Ultralytics 自己判断
DEVICE = 0
# ===============================


def check_paths():
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(f"没找到 dataset.yaml: {data_path}")

    project_path = Path(PROJECT_DIR)
    project_path.mkdir(parents=True, exist_ok=True)


def print_env_info():
    print("===== Environment Check =====")
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"cuda device {i}: {torch.cuda.get_device_name(i)}")

    has_xpu = hasattr(torch, "xpu")
    xpu_available = has_xpu and torch.xpu.is_available()
    print("xpu available:", xpu_available)
    print("=============================\n")


def main():
    check_paths()
    print_env_info()

    print("Loading model...")
    model = YOLO(MODEL_FILE)
    print("Model loaded.\n")

    train_kwargs = dict(
        data=DATA_FILE,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
    )

    if DEVICE is not None:
        train_kwargs["device"] = DEVICE

    print("Start training...")
    results = model.train(**train_kwargs)

    print("\n训练完成")
    print(f"结果目录: {PROJECT_DIR}\\{RUN_NAME}")
    print(f"best.pt 通常在: {PROJECT_DIR}\\{RUN_NAME}\\weights\\best.pt")
    print(results)


if __name__ == "__main__":
    main()
