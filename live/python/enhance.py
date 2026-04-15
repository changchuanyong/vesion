from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


# ======================
# 你只改这里
# ======================
ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "dataset" / "live" / "latest_roi.jpg"
OUTPUT_PATH = ROOT_DIR / "dataset" / "live" / "latest_roi_enhanced.jpg"

# 是否保存中间结果，方便你看每一步
SAVE_DEBUG = True
DEBUG_DIR = OUTPUT_PATH.parent / "debug_roi_enhance"

SHOW_WINDOW = os.environ.get("VISION_PIPELINE_MODE", "0") != "1"
MAX_SHOW_W = 1200
MAX_SHOW_H = 900

# ---------- ROI 增强参数 ----------
# 1) 亮部压制：>1 会压暗高亮区域
GAMMA = 1.25

# 2) CLAHE：别太大，不然噪声会起来
CLAHE_CLIP_LIMIT = 1.8
CLAHE_TILE_GRID_SIZE = (8, 8)

# 3) 双边滤波：保边去噪
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 45
BILATERAL_SIGMA_SPACE = 45

# 4) 轻微锐化：给后续 Canny 更清楚的梯度
ENABLE_SHARPEN = True
SHARPEN_SIGMA = 1.2
SHARPEN_AMOUNT = 0.55
# ======================


def atomic_imwrite(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + "_tmp" + path.suffix)

    ok = cv2.imwrite(str(tmp_path), image)
    if not ok:
        return False

    try:
        if path.exists():
            path.unlink()
        tmp_path.replace(path)
        return True
    except OSError:
        return False


def show_keep_ratio(win_name: str, img: np.ndarray, max_w: int = MAX_SHOW_W, max_h: int = MAX_SHOW_H) -> None:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    show_w = max(1, int(w * scale))
    show_h = max(1, int(h * scale))

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, show_w, show_h)
    cv2.imshow(win_name, img)


def gamma_correction(gray: np.ndarray, gamma: float = 1.25) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    corrected = np.power(gray_f, gamma)
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    return corrected


def unsharp_mask(gray: np.ndarray, sigma: float = 1.2, amount: float = 0.55) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp


def preprocess_roi(roi_bgr: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 1) 先压亮部，减轻局部过曝影响
    gamma_img = gamma_correction(gray, gamma=GAMMA)

    # 2) 局部对比增强
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    enhanced = clahe.apply(gamma_img)

    # 3) 保边去噪
    filtered = cv2.bilateralFilter(
        enhanced,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE
    )

    # 4) 轻微锐化
    if ENABLE_SHARPEN:
        final_img = unsharp_mask(
            filtered,
            sigma=SHARPEN_SIGMA,
            amount=SHARPEN_AMOUNT
        )
    else:
        final_img = filtered.copy()

    return {
        "gray": gray,
        "gamma": gamma_img,
        "clahe": enhanced,
        "filtered": filtered,
        "final": final_img,
    }


def main():
    if not INPUT_PATH.exists():
        print(f"Input not found: {INPUT_PATH}")
        return

    roi = cv2.imread(str(INPUT_PATH))
    if roi is None or roi.size == 0:
        print("Failed to read ROI image.")
        return

    results = preprocess_roi(roi)
    final_img = results["final"]

    ok = atomic_imwrite(OUTPUT_PATH, final_img)
    if not ok:
        print(f"Failed to save enhanced ROI: {OUTPUT_PATH}")
        return

    print("===== ROI Enhance Done =====")
    print(f"Input  : {INPUT_PATH}")
    print(f"Output : {OUTPUT_PATH}")
    print(f"Shape  : {roi.shape}")
    print("")
    print("Params:")
    print(f"  GAMMA                  = {GAMMA}")
    print(f"  CLAHE_CLIP_LIMIT       = {CLAHE_CLIP_LIMIT}")
    print(f"  CLAHE_TILE_GRID_SIZE   = {CLAHE_TILE_GRID_SIZE}")
    print(f"  BILATERAL_D            = {BILATERAL_D}")
    print(f"  BILATERAL_SIGMA_COLOR  = {BILATERAL_SIGMA_COLOR}")
    print(f"  BILATERAL_SIGMA_SPACE  = {BILATERAL_SIGMA_SPACE}")
    print(f"  ENABLE_SHARPEN         = {ENABLE_SHARPEN}")
    print(f"  SHARPEN_SIGMA          = {SHARPEN_SIGMA}")
    print(f"  SHARPEN_AMOUNT         = {SHARPEN_AMOUNT}")

    if SAVE_DEBUG:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        atomic_imwrite(DEBUG_DIR / "01_gray.jpg", results["gray"])
        atomic_imwrite(DEBUG_DIR / "02_gamma.jpg", results["gamma"])
        atomic_imwrite(DEBUG_DIR / "03_clahe.jpg", results["clahe"])
        atomic_imwrite(DEBUG_DIR / "04_filtered.jpg", results["filtered"])
        atomic_imwrite(DEBUG_DIR / "05_final.jpg", results["final"])
        print(f"Debug saved to: {DEBUG_DIR}")

    if SHOW_WINDOW:
        show_keep_ratio("roi_input", roi)
        show_keep_ratio("01_gray", results["gray"])
        show_keep_ratio("02_gamma", results["gamma"])
        show_keep_ratio("03_clahe", results["clahe"])
        show_keep_ratio("04_filtered", results["filtered"])
        show_keep_ratio("05_final", results["final"])
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
