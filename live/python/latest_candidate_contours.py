from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


# =========================
# 你先改这里
# =========================
ROI_IMAGE_PATH = Path(r"C:\Users\chang\Desktop\doc\vesion\dataset\live\latest_roi_enhanced.jpg")
EDGES_IMAGE_PATH = Path(r"C:\Users\chang\Desktop\doc\vesion\dataset\live\latest_edges.jpg")

OUT_DIR = ROI_IMAGE_PATH.parent
CAND_VIS_PATH = OUT_DIR / "latest_candidate_contours.jpg"
CENTER_VIS_PATH = OUT_DIR / "latest_fitted_centers.jpg"
CENTER_JSON_PATH = OUT_DIR / "latest_fitted_centers.json"

SHOW_WINDOW = True
SHOW_RAW_CENTER = True

# ---------- 单轮廓几何约束 ----------
MIN_CONTOUR_AREA = 60
MAX_CONTOUR_AREA_RATIO = 0.22

MIN_CIRCULARITY = 0.15
MAX_ASPECT_RATIO = 4.00
MIN_FILL_RATIO = 0.05

# ---------- 粗结构先验约束 ----------
BORDER_MARGIN_RATIO = 0.03
PRIOR_RX_RATIO = 0.40
PRIOR_RY_RATIO = 0.33
PRIOR_VALUE_MAX = 0.90

# ---------- 去重 ----------
DUP_CENTER_DIST = 8
MAX_KEEP = 16

# ---------- 标准孔位布局约束 ----------
ENABLE_LAYOUT_PRIOR = True

# 主锚点（两个大孔）搜索
MAIN_PAIR_MIN_DIST_RATIO = 0.18
MAIN_PAIR_MIN_HORIZONTAL = 0.55
MAIN_PAIR_MAX_LEVEL_DIFF_RATIO = 0.45
MAIN_PAIR_MIN_AREA_SIM = 0.18

# 最终分数融合
LAYOUT_SCORE_WEIGHT = 0.55
RAW_SCORE_WEIGHT = 0.45

# ---------- 顶排保底机制 ----------
ENABLE_TOP_BACKUP = True
TOP_BACKUP_MIN_RAW_SCORE = 0.30
TOP_BACKUP_MAX_LAYOUT_DIST = 1.95
TOP_BACKUP_MAX_COUNT = 1
TOP_BACKUP_MAX_ABS_U = 0.70
TOP_BACKUP_MIN_NEG_V = -0.22

# ---------- 2.3.4 椭圆拟合中心定位 ----------
FIT_MIN_POINTS = 20
FIT_MIN_POINTS_HARD = 5
FIT_MAX_ASPECT_RATIO = 5.0
FIT_MIN_AXIS = 3.0
FIT_CENTER_IN_BBOX_MARGIN = 0.35

# ---------- 显示 ----------
MAX_SHOW_W = 1000
MAX_SHOW_H = 800

PRINT_REJECT_LOG = True
# =========================


@dataclass
class CandidateContour:
    contour: np.ndarray
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    fill_ratio: float
    prior_value: float
    inside_prior: bool
    score: float

    layout_name: str = ""
    layout_group: str = ""
    layout_u: float = 0.0
    layout_v: float = 0.0
    layout_dist: float = 999.0
    layout_score: float = 0.0
    layout_accept_dist: float = 999.0
    is_main_anchor: bool = False
    is_top_backup: bool = False

    # ---------- 2.3.4 拟合结果 ----------
    fitted_center: tuple[float, float] | None = None
    fit_method: str = "none"              # ellipse / moments
    ellipse_axes: tuple[float, float] | None = None
    ellipse_angle: float | None = None


@dataclass
class LayoutModel:
    origin: tuple[float, float]
    ux: tuple[float, float]
    uy: tuple[float, float]
    scale: float
    left_anchor: CandidateContour
    right_anchor: CandidateContour


# 标准孔位布局（以两个大孔中心连线为 x 轴，孔距 d 为尺度）
# 顶排单独放宽 tol_u / tol_v / accept_dist
STANDARD_LAYOUT_ZONES = [
    # name,         u,      v,      tol_u, tol_v, group,   accept_dist
    ("top_left",    -0.30, -0.56,   0.34,  0.30, "top",    1.55),
    ("top_mid",      0.00, -0.54,   0.30,  0.28, "top",    1.48),
    ("top_right",    0.30, -0.56,   0.34,  0.30, "top",    1.55),

    ("center",       0.00, -0.20,   0.18,  0.18, "mid",    1.15),

    ("bottom_left", -0.28,  0.44,   0.24,  0.22, "bottom", 1.22),
    ("bottom_mid",   0.00,  0.62,   0.22,  0.22, "bottom", 1.25),
    ("bottom_right", 0.28,  0.44,   0.24,  0.22, "bottom", 1.22),
]


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


def atomic_write_json(path: Path, data) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if path.exists():
            path.unlink()
        tmp.replace(path)
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


def prepare_edges(edges: np.ndarray) -> np.ndarray:
    if len(edges.shape) == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    return binary


def contour_center(cnt: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) > 1e-6:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

    x, y, w, h = cv2.boundingRect(cnt)
    return x + w / 2.0, y + h / 2.0


def is_near_border(bbox: tuple[int, int, int, int], roi_shape: tuple[int, int], margin_ratio: float) -> bool:
    h_img, w_img = roi_shape[:2]
    x, y, w, h = bbox

    mx = int(w_img * margin_ratio)
    my = int(h_img * margin_ratio)

    return (x <= mx) or (y <= my) or (x + w >= w_img - mx) or (y + h >= h_img - my)


def inside_structure_prior(cx: float, cy: float, roi_shape: tuple[int, int]) -> tuple[bool, float]:
    h_img, w_img = roi_shape[:2]
    roi_cx = w_img / 2.0
    roi_cy = h_img / 2.0

    rx = PRIOR_RX_RATIO * w_img
    ry = PRIOR_RY_RATIO * h_img

    dx = cx - roi_cx
    dy = cy - roi_cy

    value = (dx * dx) / (rx * rx + 1e-6) + (dy * dy) / (ry * ry + 1e-6)
    return value <= 1.0, value


def compute_candidate_features(cnt: np.ndarray, roi_shape: tuple[int, int]) -> CandidateContour | None:
    area = float(cv2.contourArea(cnt))
    if area <= 1:
        return None

    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter <= 1:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 1 or h <= 1:
        return None

    cx, cy = contour_center(cnt)

    circularity = 4.0 * math.pi * area / (perimeter * perimeter + 1e-6)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    fill_ratio = area / (w * h + 1e-6)

    inside, prior_value = inside_structure_prior(cx, cy, roi_shape)

    roi_h, roi_w = roi_shape[:2]
    roi_area = roi_h * roi_w
    area_norm = area / (roi_area + 1e-6)

    if area_norm < 0.0015:
        area_score = 0.05
    elif area_norm < 0.004:
        area_score = 0.50
    elif area_norm < 0.035:
        area_score = 1.00
    else:
        area_score = 0.50

    center_score = max(0.0, min(1.0, 1.0 - prior_value))
    circularity_score = max(0.0, min(1.0, circularity))
    aspect_score = max(0.0, min(1.0, 1.0 / aspect_ratio))
    fill_score = max(0.0, min(1.0, fill_ratio))

    score = (
        0.25 * circularity_score +
        0.10 * aspect_score +
        0.10 * fill_score +
        0.30 * center_score +
        0.25 * area_score
    )

    return CandidateContour(
        contour=cnt,
        center=(cx, cy),
        bbox=(x, y, w, h),
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        fill_ratio=fill_ratio,
        prior_value=prior_value,
        inside_prior=inside,
        score=score,
    )


def is_duplicate_candidate(c1: CandidateContour, c2: CandidateContour, center_dist_thresh: float) -> bool:
    d = math.hypot(c1.center[0] - c2.center[0], c1.center[1] - c2.center[1])
    return d < center_dist_thresh


def reject_log(reason: str, feat: CandidateContour) -> None:
    if not PRINT_REJECT_LOG:
        return

    print(
        f"reject={reason:>16s}  "
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"area={feat.area:7.1f}  "
        f"circ={feat.circularity:5.3f}  "
        f"aspect={feat.aspect_ratio:4.2f}  "
        f"fill={feat.fill_ratio:4.2f}  "
        f"prior={feat.prior_value:4.2f}  "
        f"score={feat.score:5.3f}"
    )


def keep_log(tag: str, feat: CandidateContour) -> None:
    print(
        f"{tag:<20s}"
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"area={feat.area:7.1f}  "
        f"circ={feat.circularity:5.3f}  "
        f"aspect={feat.aspect_ratio:4.2f}  "
        f"fill={feat.fill_ratio:4.2f}  "
        f"prior={feat.prior_value:4.2f}  "
        f"score={feat.score:5.3f}"
    )


def layout_reject_log(reason: str, feat: CandidateContour) -> None:
    print(
        f"reject={reason:>16s}  "
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"uv=({feat.layout_u:5.2f},{feat.layout_v:5.2f})  "
        f"layout={feat.layout_name:<12s}  "
        f"group={feat.layout_group:<7s}  "
        f"dist={feat.layout_dist:5.2f}/{feat.layout_accept_dist:4.2f}  "
        f"raw={feat.score:5.3f}  "
        f"final={feat.layout_score:5.3f}"
    )


def layout_keep_log(tag: str, feat: CandidateContour) -> None:
    print(
        f"{tag:<20s}"
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"uv=({feat.layout_u:5.2f},{feat.layout_v:5.2f})  "
        f"layout={feat.layout_name:<12s}  "
        f"group={feat.layout_group:<7s}  "
        f"dist={feat.layout_dist:5.2f}/{feat.layout_accept_dist:4.2f}  "
        f"raw={feat.score:5.3f}  "
        f"final={feat.layout_score:5.3f}"
    )


def find_main_big_hole_pair(candidates: list[CandidateContour], roi_shape: tuple[int, int]) -> tuple[int, int] | None:
    if len(candidates) < 2:
        return None

    h_img, w_img = roi_shape[:2]
    roi_area = h_img * w_img
    min_dim = min(h_img, w_img)

    best_pair = None
    best_score = -1.0

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1 = candidates[i]
            c2 = candidates[j]

            dx = c2.center[0] - c1.center[0]
            dy = c2.center[1] - c1.center[1]
            dist = math.hypot(dx, dy)
            if dist < MAIN_PAIR_MIN_DIST_RATIO * min_dim:
                continue

            horizontal = abs(dx) / (dist + 1e-6)
            if horizontal < MAIN_PAIR_MIN_HORIZONTAL:
                continue

            level_diff_ratio = abs(dy) / (dist + 1e-6)
            if level_diff_ratio > MAIN_PAIR_MAX_LEVEL_DIFF_RATIO:
                continue

            area_sim = min(c1.area, c2.area) / (max(c1.area, c2.area) + 1e-6)
            if area_sim < MAIN_PAIR_MIN_AREA_SIM:
                continue

            area_sum_norm = min(1.0, (c1.area + c2.area) / (0.05 * roi_area + 1e-6))
            center_prior = 1.0 - min(1.0, (c1.prior_value + c2.prior_value) / 2.0)

            pair_score = (
                0.45 * area_sum_norm +
                0.20 * area_sim +
                0.20 * horizontal +
                0.10 * (1.0 - level_diff_ratio) +
                0.05 * center_prior
            )

            if pair_score > best_score:
                best_score = pair_score
                best_pair = (i, j)

    return best_pair


def build_layout_model(left_anchor: CandidateContour, right_anchor: CandidateContour) -> LayoutModel | None:
    p1 = np.array(left_anchor.center, dtype=np.float32)
    p2 = np.array(right_anchor.center, dtype=np.float32)

    vec = p2 - p1
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return None

    ux = vec / dist
    uy = np.array([-ux[1], ux[0]], dtype=np.float32)

    origin = (p1 + p2) / 2.0

    return LayoutModel(
        origin=(float(origin[0]), float(origin[1])),
        ux=(float(ux[0]), float(ux[1])),
        uy=(float(uy[0]), float(uy[1])),
        scale=dist,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
    )


def project_candidate_to_layout(cand: CandidateContour, model: LayoutModel) -> tuple[float, float]:
    p = np.array(cand.center, dtype=np.float32)
    origin = np.array(model.origin, dtype=np.float32)
    ux = np.array(model.ux, dtype=np.float32)
    uy = np.array(model.uy, dtype=np.float32)

    delta = p - origin
    u = float(np.dot(delta, ux) / (model.scale + 1e-6))
    v = float(np.dot(delta, uy) / (model.scale + 1e-6))
    return u, v


def layout_uv_to_xy(u: float, v: float, model: LayoutModel) -> tuple[int, int]:
    origin = np.array(model.origin, dtype=np.float32)
    ux = np.array(model.ux, dtype=np.float32)
    uy = np.array(model.uy, dtype=np.float32)
    p = origin + model.scale * (u * ux + v * uy)
    return int(round(float(p[0]))), int(round(float(p[1])))


def match_candidate_to_layout_zone(cand: CandidateContour, model: LayoutModel) -> CandidateContour:
    u, v = project_candidate_to_layout(cand, model)
    cand.layout_u = u
    cand.layout_v = v

    best_name = ""
    best_group = ""
    best_dist = 999.0
    best_accept = 999.0

    for zone_name, z_u, z_v, tol_u, tol_v, zone_group, accept_dist in STANDARD_LAYOUT_ZONES:
        du = (u - z_u) / (tol_u + 1e-6)
        dv = (v - z_v) / (tol_v + 1e-6)
        d = math.hypot(du, dv)

        if d < best_dist:
            best_dist = d
            best_name = zone_name
            best_group = zone_group
            best_accept = accept_dist

    cand.layout_name = best_name
    cand.layout_group = best_group
    cand.layout_dist = best_dist
    cand.layout_accept_dist = best_accept

    closeness = max(0.0, 1.0 - best_dist / (best_accept + 1e-6))
    cand.layout_score = LAYOUT_SCORE_WEIGHT * closeness + RAW_SCORE_WEIGHT * cand.score
    return cand


def apply_standard_layout_prior(
    candidates: list[CandidateContour],
    roi_shape: tuple[int, int]
) -> tuple[list[CandidateContour], LayoutModel | None]:
    if not ENABLE_LAYOUT_PRIOR:
        return candidates, None

    if len(candidates) < 2:
        return candidates, None

    print("\n===== Stage 4: standard layout prior filtering =====")

    pair_idx = find_main_big_hole_pair(candidates, roi_shape)
    if pair_idx is None:
        print("No valid main-hole pair found. Skip layout prior filtering.")
        return candidates, None

    c1 = candidates[pair_idx[0]]
    c2 = candidates[pair_idx[1]]

    if c1.center[0] <= c2.center[0]:
        left_anchor, right_anchor = c1, c2
    else:
        left_anchor, right_anchor = c2, c1

    model = build_layout_model(left_anchor, right_anchor)
    if model is None:
        print("Failed to build layout model. Skip layout prior filtering.")
        return candidates, None

    left_anchor.is_main_anchor = True
    right_anchor.is_main_anchor = True
    left_anchor.layout_name = "main_left"
    right_anchor.layout_name = "main_right"
    left_anchor.layout_group = "anchor"
    right_anchor.layout_group = "anchor"
    left_anchor.layout_u, left_anchor.layout_v = -0.5, 0.0
    right_anchor.layout_u, right_anchor.layout_v = 0.5, 0.0
    left_anchor.layout_dist = 0.0
    right_anchor.layout_dist = 0.0
    left_anchor.layout_score = left_anchor.score
    right_anchor.layout_score = right_anchor.score
    left_anchor.layout_accept_dist = 0.0
    right_anchor.layout_accept_dist = 0.0

    print(
        f"main_pair left=({left_anchor.center[0]:.1f}, {left_anchor.center[1]:.1f})  "
        f"right=({right_anchor.center[0]:.1f}, {right_anchor.center[1]:.1f})  "
        f"dist={model.scale:.1f}"
    )

    zone_best: dict[str, CandidateContour] = {}
    top_backups: list[CandidateContour] = []

    others = [c for c in candidates if c is not left_anchor and c is not right_anchor]

    for cand in others:
        match_candidate_to_layout_zone(cand, model)

        if cand.layout_dist <= cand.layout_accept_dist:
            old = zone_best.get(cand.layout_name)
            if old is None:
                zone_best[cand.layout_name] = cand
                layout_keep_log("keep_layout_raw", cand)
            else:
                if cand.layout_score > old.layout_score:
                    layout_reject_log("layout_replaced", old)
                    zone_best[cand.layout_name] = cand
                    layout_keep_log("keep_layout_best", cand)
                else:
                    layout_reject_log("layout_weaker", cand)
            continue

        if (
            ENABLE_TOP_BACKUP
            and cand.layout_group == "top"
            and cand.layout_v < TOP_BACKUP_MIN_NEG_V
            and abs(cand.layout_u) < TOP_BACKUP_MAX_ABS_U
            and cand.score >= TOP_BACKUP_MIN_RAW_SCORE
            and cand.layout_dist <= TOP_BACKUP_MAX_LAYOUT_DIST
        ):
            cand.is_top_backup = True
            cand.layout_score = 0.20 + 0.80 * cand.score
            top_backups.append(cand)
            layout_keep_log("keep_top_backup", cand)
            continue

        layout_reject_log("layout_outlier", cand)

    final_kept: list[CandidateContour] = [left_anchor, right_anchor]

    zone_order = [z[0] for z in STANDARD_LAYOUT_ZONES]
    for name in zone_order:
        if name in zone_best:
            final_kept.append(zone_best[name])

    kept_top_count = sum(1 for c in final_kept if c.layout_group == "top")

    if kept_top_count < 2 and len(top_backups) > 0:
        top_backups.sort(key=lambda c: c.layout_score, reverse=True)

        added = 0
        for cand in top_backups:
            too_close = False
            for old in final_kept:
                if is_duplicate_candidate(cand, old, DUP_CENTER_DIST):
                    too_close = True
                    break

            if too_close:
                continue

            final_kept.append(cand)
            added += 1
            if added >= TOP_BACKUP_MAX_COUNT:
                break

    def sort_key(c: CandidateContour):
        if c.is_main_anchor:
            return (-10.0, c.center[0])
        return (c.layout_v, c.layout_u)

    final_kept.sort(key=sort_key)

    print("\n===== Layout final kept =====")
    for cand in final_kept:
        if cand.is_main_anchor:
            print(
                f"anchor                center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"layout={cand.layout_name:<12s} raw={cand.score:5.3f}"
            )
        else:
            extra = " backup" if cand.is_top_backup else ""
            print(
                f"layout_keep{extra:<7s}  "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"uv=({cand.layout_u:5.2f},{cand.layout_v:5.2f})  "
                f"layout={cand.layout_name:<12s}  "
                f"group={cand.layout_group:<7s}  "
                f"dist={cand.layout_dist:5.2f}/{cand.layout_accept_dist:4.2f}  "
                f"raw={cand.score:5.3f}  "
                f"final={cand.layout_score:5.3f}"
            )

    return final_kept, model


def filter_candidate_contours(
    edges: np.ndarray,
    roi_bgr: np.ndarray
) -> tuple[list[CandidateContour], list[np.ndarray], LayoutModel | None]:
    h_img, w_img = roi_bgr.shape[:2]
    roi_area = h_img * w_img
    max_area = MAX_CONTOUR_AREA_RATIO * roi_area

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    raw_candidates: list[CandidateContour] = []

    print("===== Stage 1&2: geometry + coarse prior filtering =====")
    for cnt in contours:
        feat = compute_candidate_features(cnt, roi_bgr.shape)
        if feat is None:
            continue

        if feat.area < MIN_CONTOUR_AREA:
            reject_log("area_small", feat)
            continue

        if feat.area > max_area:
            reject_log("area_large", feat)
            continue

        if feat.circularity < MIN_CIRCULARITY:
            reject_log("circularity", feat)
            continue

        if feat.aspect_ratio > MAX_ASPECT_RATIO:
            reject_log("aspect", feat)
            continue

        if feat.fill_ratio < MIN_FILL_RATIO:
            reject_log("fill_ratio", feat)
            continue

        if is_near_border(feat.bbox, roi_bgr.shape, BORDER_MARGIN_RATIO):
            reject_log("near_border", feat)
            continue

        if not feat.inside_prior:
            reject_log("outside_prior", feat)
            continue

        if feat.prior_value > PRIOR_VALUE_MAX:
            reject_log("prior_too_far", feat)
            continue

        raw_candidates.append(feat)
        keep_log("keep_raw", feat)

    raw_candidates.sort(key=lambda c: c.score, reverse=True)

    print("\n===== Stage 3: duplicate removal =====")
    kept: list[CandidateContour] = []
    for cand in raw_candidates:
        duplicated = False
        for old in kept:
            if is_duplicate_candidate(cand, old, DUP_CENTER_DIST):
                print(
                    f"reject=duplicate       "
                    f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                    f"score={cand.score:5.3f}  "
                    f"close_to=({old.center[0]:6.1f},{old.center[1]:6.1f})"
                )
                duplicated = True
                break

        if not duplicated:
            kept.append(cand)
            keep_log("keep_final_dup", cand)

        if len(kept) >= MAX_KEEP:
            break

    kept_layout, layout_model = apply_standard_layout_prior(kept, roi_bgr.shape)
    return kept_layout, contours, layout_model


# =========================
# 2.3.4 只对 kept candidates 做椭圆拟合
# =========================
def ellipse_center_valid(
    center: tuple[float, float],
    bbox: tuple[int, int, int, int],
    margin_ratio: float,
) -> bool:
    cx, cy = center
    x, y, w, h = bbox

    mx = w * margin_ratio
    my = h * margin_ratio

    return (x - mx <= cx <= x + w + mx) and (y - my <= cy <= y + h + my)


def fit_ellipse_on_kept_candidates(
    kept_candidates: list[CandidateContour],
) -> list[CandidateContour]:
    print("\n===== Stage 5: fitEllipse on kept candidates =====")

    fitted_results: list[CandidateContour] = []

    for cand in kept_candidates:
        cnt = cand.contour
        n_points = len(cnt)

        cand.fitted_center = cand.center
        cand.fit_method = "moments"
        cand.ellipse_axes = None
        cand.ellipse_angle = None

        if n_points < FIT_MIN_POINTS_HARD:
            print(
                f"fit=fallback_points   "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"points={n_points}"
            )
            fitted_results.append(cand)
            continue

        if n_points < FIT_MIN_POINTS:
            print(
                f"fit=fallback_sparse   "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"points={n_points}"
            )
            fitted_results.append(cand)
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            print(
                f"fit=fallback_error    "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})"
            )
            fitted_results.append(cand)
            continue

        (cx, cy), (ma, mi), angle = ellipse

        if ma < FIT_MIN_AXIS or mi < FIT_MIN_AXIS:
            print(
                f"fit=fallback_axis     "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"axes=({ma:.1f},{mi:.1f})"
            )
            fitted_results.append(cand)
            continue

        ellipse_aspect_ratio = max(ma, mi) / (min(ma, mi) + 1e-6)
        if ellipse_aspect_ratio > FIT_MAX_ASPECT_RATIO:
            print(
                f"fit=fallback_aspect   "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"e_aspect={ellipse_aspect_ratio:.2f}"
            )
            fitted_results.append(cand)
            continue

        if not ellipse_center_valid((cx, cy), cand.bbox, FIT_CENTER_IN_BBOX_MARGIN):
            print(
                f"fit=fallback_bbox     "
                f"raw=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"fit=({cx:6.1f},{cy:6.1f})"
            )
            fitted_results.append(cand)
            continue

        cand.fitted_center = (float(cx), float(cy))
        cand.fit_method = "ellipse"
        cand.ellipse_axes = (float(ma), float(mi))
        cand.ellipse_angle = float(angle)

        print(
            f"fit=ellipse           "
            f"raw=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
            f"fit=({cx:6.1f},{cy:6.1f})  "
            f"axes=({ma:5.1f},{mi:5.1f})  "
            f"angle={angle:6.1f}"
        )

        fitted_results.append(cand)

    return fitted_results


def draw_layout_prior(vis: np.ndarray, model: LayoutModel) -> None:
    p1 = (int(round(model.left_anchor.center[0])), int(round(model.left_anchor.center[1])))
    p2 = (int(round(model.right_anchor.center[0])), int(round(model.right_anchor.center[1])))
    cv2.line(vis, p1, p2, (0, 255, 255), 1)

    oc = (int(round(model.origin[0])), int(round(model.origin[1])))
    cv2.circle(vis, oc, 3, (255, 255, 0), -1)

    for zone_name, z_u, z_v, tol_u, tol_v, zone_group, accept_dist in STANDARD_LAYOUT_ZONES:
        pt = layout_uv_to_xy(z_u, z_v, model)
        cv2.circle(vis, pt, 4, (255, 0, 255), 1)
        cv2.putText(
            vis,
            zone_name,
            (pt[0] + 4, pt[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 255),
            1,
        )

        pt1 = layout_uv_to_xy(z_u - tol_u, z_v - tol_v, model)
        pt2 = layout_uv_to_xy(z_u + tol_u, z_v + tol_v, model)
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(
            vis,
            (min(x1, x2), min(y1, y2)),
            (max(x1, x2), max(y1, y2)),
            (200, 80, 200),
            1,
        )


def build_candidate_vis(
    roi_bgr: np.ndarray,
    all_contours: list[np.ndarray],
    kept_candidates: list[CandidateContour],
    layout_model: LayoutModel | None,
) -> np.ndarray:
    vis = roi_bgr.copy()
    h_img, w_img = roi_bgr.shape[:2]

    cv2.drawContours(vis, all_contours, -1, (160, 160, 160), 1)

    center = (int(w_img / 2), int(h_img / 2))
    axes = (int(PRIOR_RX_RATIO * w_img), int(PRIOR_RY_RATIO * h_img))
    cv2.ellipse(vis, center, axes, 0, 0, 360, (255, 255, 0), 1)
    cv2.circle(vis, center, 3, (255, 255, 0), -1)

    if layout_model is not None:
        draw_layout_prior(vis, layout_model)

    for i, cand in enumerate(kept_candidates):
        if cand.is_main_anchor:
            contour_color = (0, 165, 255)
            center_color = (0, 0, 255)
            box_color = (0, 165, 255)
        elif cand.is_top_backup:
            contour_color = (255, 180, 0)
            center_color = (0, 0, 255)
            box_color = (255, 180, 0)
        else:
            contour_color = (0, 255, 0)
            center_color = (0, 0, 255)
            box_color = (255, 0, 0)

        cv2.drawContours(vis, [cand.contour], -1, contour_color, 2)

        cx = int(round(cand.center[0]))
        cy = int(round(cand.center[1]))
        cv2.circle(vis, (cx, cy), 3, center_color, -1)

        x, y, w, h = cand.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, 1)

        if cand.is_main_anchor:
            text = f"{i}: {cand.layout_name} s={cand.score:.2f}"
        elif cand.is_top_backup:
            text = f"{i}: top_backup d={cand.layout_dist:.2f} s={cand.layout_score:.2f}"
        else:
            text = f"{i}: {cand.layout_name} d={cand.layout_dist:.2f} s={cand.layout_score:.2f}"

        cv2.putText(
            vis,
            text,
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
        )

    return vis


def build_center_vis(
    roi_bgr: np.ndarray,
    fitted_candidates: list[CandidateContour],
) -> np.ndarray:
    vis = roi_bgr.copy()

    for i, cand in enumerate(fitted_candidates):
        if cand.is_main_anchor:
            contour_color = (0, 165, 255)
            box_color = (0, 165, 255)
        elif cand.is_top_backup:
            contour_color = (255, 180, 0)
            box_color = (255, 180, 0)
        else:
            contour_color = (0, 255, 0)
            box_color = (255, 0, 0)

        cv2.drawContours(vis, [cand.contour], -1, contour_color, 2)

        x, y, w, h = cand.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, 1)

        # 原始中心：紫色
        if SHOW_RAW_CENTER:
            raw_cx = int(round(cand.center[0]))
            raw_cy = int(round(cand.center[1]))
            cv2.circle(vis, (raw_cx, raw_cy), 3, (255, 0, 255), -1)

        # 拟合中心：红色
        if cand.fitted_center is not None:
            fit_cx = int(round(cand.fitted_center[0]))
            fit_cy = int(round(cand.fitted_center[1]))
            cv2.circle(vis, (fit_cx, fit_cy), 4, (0, 0, 255), -1)

            if SHOW_RAW_CENTER:
                cv2.line(vis, (raw_cx, raw_cy), (fit_cx, fit_cy), (255, 0, 255), 1)

        # 拟合椭圆：黄色
        if cand.fit_method == "ellipse" and cand.fitted_center is not None and cand.ellipse_axes is not None and cand.ellipse_angle is not None:
            ellipse = (
                (float(cand.fitted_center[0]), float(cand.fitted_center[1])),
                (float(cand.ellipse_axes[0]), float(cand.ellipse_axes[1])),
                float(cand.ellipse_angle),
            )
            cv2.ellipse(vis, ellipse, (0, 255, 255), 2)

        text = f"{i}: {cand.fit_method}"
        cv2.putText(
            vis,
            text,
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    return vis


def export_fitted_centers(fitted_candidates: list[CandidateContour]) -> list[dict]:
    out = []
    for i, c in enumerate(fitted_candidates):
        item = {
            "index": i,
            "raw_center": [round(c.center[0], 3), round(c.center[1], 3)],
            "fitted_center": None if c.fitted_center is None else [round(c.fitted_center[0], 3), round(c.fitted_center[1], 3)],
            "fit_method": c.fit_method,
            "bbox": [int(c.bbox[0]), int(c.bbox[1]), int(c.bbox[2]), int(c.bbox[3])],
            "area": round(c.area, 3),
            "score": round(c.score, 6),
            "layout_name": c.layout_name,
            "layout_group": c.layout_group,
            "layout_dist": round(c.layout_dist, 6),
            "layout_accept_dist": round(c.layout_accept_dist, 6),
            "is_main_anchor": c.is_main_anchor,
            "is_top_backup": c.is_top_backup,
            "ellipse_axes": None if c.ellipse_axes is None else [round(c.ellipse_axes[0], 3), round(c.ellipse_axes[1], 3)],
            "ellipse_angle": None if c.ellipse_angle is None else round(c.ellipse_angle, 3),
        }
        out.append(item)
    return out


def main():
    if not ROI_IMAGE_PATH.exists():
        print(f"ROI not found: {ROI_IMAGE_PATH}")
        return

    if not EDGES_IMAGE_PATH.exists():
        print(f"Edges not found: {EDGES_IMAGE_PATH}")
        return

    roi = cv2.imread(str(ROI_IMAGE_PATH))
    edges = cv2.imread(str(EDGES_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    if roi is None or roi.size == 0:
        print("Failed to read ROI image.")
        return

    if edges is None or edges.size == 0:
        print("Failed to read edges image.")
        return

    edges_bin = prepare_edges(edges)

    print("===== Input =====")
    print(f"ROI path        : {ROI_IMAGE_PATH}")
    print(f"Edges path      : {EDGES_IMAGE_PATH}")
    print(f"Candidate vis   : {CAND_VIS_PATH}")
    print(f"Center vis      : {CENTER_VIS_PATH}")
    print(f"Center json     : {CENTER_JSON_PATH}")
    print(f"ROI shape       : {roi.shape}")
    print(f"Edges shape     : {edges_bin.shape}\n")

    # 2.3.3 候选筛选
    candidates, all_contours, layout_model = filter_candidate_contours(edges_bin, roi)

    # 2.3.3 可视化
    cand_vis = build_candidate_vis(roi, all_contours, candidates, layout_model)
    atomic_imwrite(CAND_VIS_PATH, cand_vis)

    # 2.3.4 只对已保留候选做椭圆拟合
    fitted_candidates = fit_ellipse_on_kept_candidates(candidates)

    # 2.3.4 可视化与导出
    center_vis = build_center_vis(roi, fitted_candidates)
    atomic_imwrite(CENTER_VIS_PATH, center_vis)

    center_data = export_fitted_centers(fitted_candidates)
    atomic_write_json(CENTER_JSON_PATH, center_data)

    print("\n===== Final kept candidates =====")
    print(f"Kept candidates: {len(candidates)}")
    for i, c in enumerate(candidates):
        if c.is_main_anchor:
            print(
                f"[{i}] "
                f"center=({c.center[0]:.1f}, {c.center[1]:.1f})  "
                f"area={c.area:.1f}  "
                f"anchor={c.layout_name}  "
                f"raw_score={c.score:.3f}"
            )
        else:
            extra = " top_backup" if c.is_top_backup else ""
            print(
                f"[{i}] "
                f"center=({c.center[0]:.1f}, {c.center[1]:.1f})  "
                f"area={c.area:.1f}  "
                f"layout={c.layout_name}{extra}  "
                f"uv=({c.layout_u:.2f}, {c.layout_v:.2f})  "
                f"dist={c.layout_dist:.2f}/{c.layout_accept_dist:.2f}  "
                f"raw={c.score:.3f}  "
                f"final={c.layout_score:.3f}"
            )

    print("\n===== Final fitted centers =====")
    for i, c in enumerate(fitted_candidates):
        if c.fitted_center is None:
            continue

        print(
            f"[{i}] "
            f"raw_center=({c.center[0]:.1f}, {c.center[1]:.1f})  "
            f"fit_center=({c.fitted_center[0]:.1f}, {c.fitted_center[1]:.1f})  "
            f"method={c.fit_method}"
        )

    print(f"\nSaved candidate vis: {CAND_VIS_PATH}")
    print(f"Saved center vis   : {CENTER_VIS_PATH}")
    print(f"Saved center json  : {CENTER_JSON_PATH}")

    if SHOW_WINDOW:
        show_keep_ratio("roi", roi)
        show_keep_ratio("edges", edges_bin)
        show_keep_ratio("candidate_contours", cand_vis)
        show_keep_ratio("fitted_centers", center_vis)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()