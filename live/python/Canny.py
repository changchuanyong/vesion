from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


# =========================
# 你先改这里
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
ROI_IMAGE_PATH = ROOT_DIR / "dataset" / "live" / "latest_roi_enhanced.jpg"
OUT_DIR = ROI_IMAGE_PATH.parent

EDGES_PATH = OUT_DIR / "latest_edges.jpg"
VIS_PATH = OUT_DIR / "latest_edges_vis.jpg"

SHOW_WINDOW = True
SLEEP_SHORT = 0.05

# 显示窗口最大尺寸（按比例缩放，不拉伸）
MAX_SHOW_W = 900
MAX_SHOW_H = 700

# 预处理参数
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50

# 自适应 Canny 参数
# 推荐先用这个范围：
# HIGH_PCT 80~90
# LOW_RATIO 0.35~0.55
HIGH_PCT = 90
LOW_RATIO = 0.35

# 形态学
CLOSE_KERNEL = 6
CLOSE_ITER = 1

# 小连通域去除
MIN_COMPONENT_AREA = 20
# =========================


def atomic_imwrite(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)

    ok = cv2.imwrite(str(tmp), image)
    if not ok:
        return False

    try:
        if path.exists():
            path.unlink()
        tmp.replace(path)
        return True
    except OSError:
        return False


def remove_small_components(binary_img: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    out = np.zeros_like(binary_img)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255

    return out


def preprocess(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    enhanced = clahe.apply(gray)

    smooth = cv2.bilateralFilter(
        enhanced,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )
    return enhanced, smooth


def compute_adaptive_thresholds(img_gray: np.ndarray) -> tuple[int, int, np.ndarray]:
    """
    用梯度幅值统计来估计 Canny 阈值：
    先对预处理后的灰度图求 Sobel 梯度，
    再按梯度分布百分位数自适应给出 high / low。
    """
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    mag_u8 = cv2.convertScaleAbs(mag)

    valid = mag_u8[mag_u8 > 0]
    if valid.size < 50:
        return 30, 80, mag_u8

    high = int(np.percentile(valid, HIGH_PCT))
    high = max(20, min(255, high))

    low = int(max(0, min(high - 1, LOW_RATIO * high)))
    low = max(5, low)

    return low, high, mag_u8


def adaptive_canny_pipeline(roi_bgr: np.ndarray):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    enhanced, smooth = preprocess(gray)
    low, high, grad_mag = compute_adaptive_thresholds(smooth)

    edges = cv2.Canny(
        smooth,
        threshold1=low,
        threshold2=high,
        L2gradient=True
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITER)

    edges = remove_small_components(edges, MIN_COMPONENT_AREA)

    return gray, enhanced, smooth, grad_mag, edges, low, high


def build_vis(roi_bgr: np.ndarray, edges: np.ndarray, low: int, high: int) -> np.ndarray:
    vis = roi_bgr.copy()

    # 红色叠加边缘
    vis[edges > 0] = (0, 0, 255)

    cv2.putText(
        vis,
        f"Adaptive Canny  low={low} high={high}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        f"HIGH_PCT={HIGH_PCT} LOW_RATIO={LOW_RATIO:.2f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    return vis


def show_keep_ratio(win_name: str, img: np.ndarray, max_w: int = MAX_SHOW_W, max_h: int = MAX_SHOW_H) -> None:
    """
    按原图比例显示窗口，避免 imshow 窗口被手动/系统拉伸后看起来变形。
    这里只控制显示，不改变图像实际尺寸。
    """
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    show_w = max(1, int(w * scale))
    show_h = max(1, int(h * scale))

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, show_w, show_h)
    cv2.imshow(win_name, img)


def main():
    print("Adaptive Canny watch started.")
    print(f"Watching: {ROI_IMAGE_PATH}")
    print(f"Edges out: {EDGES_PATH}")
    print(f"Vis out  : {VIS_PATH}")
    print("Press ESC to quit.")

    last_mtime = 0.0
    printed_shape = False

    while True:
        if not ROI_IMAGE_PATH.exists():
            time.sleep(SLEEP_SHORT)
            continue

        try:
            stat = ROI_IMAGE_PATH.stat()
            file_size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            time.sleep(SLEEP_SHORT)
            continue

        if file_size <= 0 or mtime == last_mtime:
            time.sleep(SLEEP_SHORT)
            continue

        frame = cv2.imread(str(ROI_IMAGE_PATH))
        if frame is None or frame.size == 0:
            time.sleep(SLEEP_SHORT)
            continue

        last_mtime = mtime

        gray, enhanced, smooth, grad_mag, edges, low, high = adaptive_canny_pipeline(frame)
        vis = build_vis(frame, edges, low, high)

        atomic_imwrite(EDGES_PATH, edges)
        atomic_imwrite(VIS_PATH, vis)

        if not printed_shape:
            print("frame.shape    =", frame.shape)
            print("enhanced.shape =", enhanced.shape)
            print("smooth.shape   =", smooth.shape)
            print("grad_mag.shape =", grad_mag.shape)
            print("edges.shape    =", edges.shape)
            printed_shape = True

        if SHOW_WINDOW:
            show_keep_ratio("roi", frame)
            show_keep_ratio("enhanced", enhanced)
            show_keep_ratio("smooth", smooth)
            show_keep_ratio("grad_mag", grad_mag)
            show_keep_ratio("edges", edges)
            show_keep_ratio("edges_vis", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
