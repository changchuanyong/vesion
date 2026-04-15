from __future__ import annotations

import time
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT_DIR / "dataset" / "live"

WATCH_IMAGES = {
    "latest_fitted_centers": LIVE_DIR / "latest_fitted_centers.jpg",
    "latest_candidate_contours": LIVE_DIR / "latest_candidate_contours.jpg",
    "latest_pose_vis": LIVE_DIR / "latest_pose_vis.jpg",
}

SLEEP_SHORT = 0.05
MAX_SHOW_W = 1000
MAX_SHOW_H = 800


def show_keep_ratio(win_name: str, img, max_w: int = MAX_SHOW_W, max_h: int = MAX_SHOW_H) -> None:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    show_w = max(1, int(w * scale))
    show_h = max(1, int(h * scale))
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, show_w, show_h)
    cv2.imshow(win_name, img)


def main():
    print("Post visualization watch started.")
    for win_name, p in WATCH_IMAGES.items():
        print(f"{win_name}: {p}")
    print("Press ESC to quit.")

    last_mtime = {k: 0.0 for k in WATCH_IMAGES.keys()}
    cached_img = {}

    while True:
        for win_name, img_path in WATCH_IMAGES.items():
            if not img_path.exists():
                continue

            try:
                stat = img_path.stat()
            except OSError:
                continue

            if stat.st_size <= 0:
                continue

            if stat.st_mtime > last_mtime[win_name]:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    continue
                last_mtime[win_name] = stat.st_mtime
                cached_img[win_name] = img

            if win_name in cached_img:
                show_keep_ratio(win_name, cached_img[win_name])

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        time.sleep(SLEEP_SHORT)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
