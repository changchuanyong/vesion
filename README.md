# Vision

基于 YOLO 的充电口视觉项目，包含：
- 标注转换（Labelme -> YOLO）
- 目标检测训练（Ultralytics YOLO）
- 实时图像处理流水线（ROI 提取、增强、边缘、轮廓、PnP 位姿）

## 项目结构

```text
Vision/
├─ train/python/
│  ├─ convert_labelme_to_yolo_detect.py   # 标注转换
│  └─ train_port.py                       # YOLO 训练
├─ live/python/
│  ├─ launch_pipeline.py                  # 实时流水线启动器
│  ├─ roi.py / enhance.py / Canny.py
│  ├─ latest_candidate_contours.py
│  ├─ pnp2.py
│  └─ post_vis_watch.py
├─ live/cpp/
│  └─ kinect_live_capture_atomic.cpp      # 相机采集（需编译为 exe）
├─ yolo_port/
│  ├─ images/train,val
│  ├─ labels/train,val
│  └─ dataset.yaml
├─ config/charging_port_model.csv         # PnP 模型点
└─ dataset/live/                          # 实时输出目录（运行时生成/更新）
```

## 环境要求

- Python 3.9+（建议用你当前的 `.yolo_env`）
- 主要依赖：
  - `ultralytics`
  - `torch`
  - `opencv-python`
  - `numpy`
- Windows（当前脚本路径和进程管理按 Windows 组织）

安装示例：

```powershell
pip install ultralytics torch opencv-python numpy
```

## 数据准备与转换

1. 把标注数据放在：
   - `yolo_port/images/train`
   - `yolo_port/images/val`
2. 运行转换脚本（会生成 `yolo_port/labels/*` 和 `yolo_port/dataset.yaml`）：

```powershell
python train\python\convert_labelme_to_yolo_detect.py
```

## 模型训练

训练入口：

```powershell
python train\python\train_port.py
```

默认配置在 `train/python/train_port.py`：
- `MODEL_FILE = "yolov8n.pt"`
- `EPOCHS = 100`
- `IMGSZ = 640`
- `BATCH = 8`
- `DEVICE = 0`（GPU）

训练输出默认在：
- `runs/detect_retrain/weights/best.pt`

## 实时流水线运行

启动脚本：

```powershell
python live\python\launch_pipeline.py
```

该脚本会串联：
1. `kinect_live_capture_atomic.exe`（采集）
2. `roi.py`（检测并裁 ROI）
3. `enhance.py`（增强）
4. `Canny.py`（边缘）
5. `latest_candidate_contours.py`（候选点/轮廓）
6. `pnp2.py`（位姿计算）
7. `post_vis_watch.py`（可视化监看）

注意：
- `live/python/launch_pipeline.py` 默认期望存在：
  - `live/cpp/kinect_live_capture_atomic.exe`
- `roi.py` 默认加载：
  - `runs/detect_retrain/weights/best.pt`

## 常见问题

- 启动后找不到模型：
  - 先完成训练，确认 `runs/detect_retrain/weights/best.pt` 存在。
- 找不到采集 exe：
  - 先编译 `live/cpp/kinect_live_capture_atomic.cpp`，并把产物放到 `live/cpp/`。
- PnP 结果异常：
  - 检查 `config/charging_port_model.csv` 的点名是否与 `latest_fitted_centers.json` 中映射一致。
  - 检查 `live/python/pnp2.py` 中相机内参 `K` 与 `DIST_COEFFS` 是否已替换为标定值。

## 备注

- 仓库中的 `runs/`、`*.pt`、`dataset/live/latest*.jpg/json` 等运行产物已在 `.gitignore` 中忽略。
