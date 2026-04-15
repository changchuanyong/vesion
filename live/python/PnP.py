from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np


# =========================
# 1. 路径
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_CSV = ROOT_DIR / "config" / "charging_port_model.csv"
FITTED_JSON = ROOT_DIR / "dataset" / "live" / "latest_fitted_centers.json"


# =========================
# 2. layout_name -> 标准标签
# =========================
LAYOUT_TO_STD = {
    "main_left": "DC_neg",
    "main_right": "DC_pos",
    "top_left": "S_neg",
    "top_mid": "CC2",
    "top_right": "S_pos",
    "center": "CC1",
    "bottom_left": "A_neg",
    "bottom_mid": "PE",
    "bottom_right": "A_pos",
}


# =========================
# 3. 读取标准模型三维点
# CSV格式:
# label,x_mm,y_mm,z_mm
# S_neg,-12,21,0
# ...
# =========================
def load_model_points_from_csv(csv_path: str | Path) -> Dict[str, np.ndarray]:
    model_points: Dict[str, np.ndarray] = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            x = float(row["x_mm"])
            y = float(row["y_mm"])
            z = float(row["z_mm"])
            model_points[label] = np.array([x, y, z], dtype=np.float64)

    if not model_points:
        raise ValueError(f"模型点文件为空: {csv_path}")

    return model_points


# =========================
# 4. 从 latest_fitted_centers.json 读取二维点
# 默认优先 fitted_center
# =========================
def load_image_points_from_fitted_json(json_path: str | Path) -> Dict[str, Tuple[float, float]]:
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    image_points_dict: Dict[str, Tuple[float, float]] = {}

    for item in data:
        layout_name = item.get("layout_name")
        if layout_name not in LAYOUT_TO_STD:
            continue

        std_name = LAYOUT_TO_STD[layout_name]

        if "fitted_center" in item and item["fitted_center"] is not None:
            cx, cy = item["fitted_center"]
        else:
            cx, cy = item["raw_center"]

        image_points_dict[std_name] = (float(cx), float(cy))

    if not image_points_dict:
        raise ValueError(f"未从 JSON 中读取到有效二维点: {json_path}")

    return image_points_dict


# =========================
# 5. 组装二维-三维对应
# 固定顺序，避免字典顺序带来问题
# =========================
def build_correspondences(
    image_points_dict: Dict[str, Tuple[float, float]],
    model_points_dict: Dict[str, np.ndarray],
    min_points: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    preferred_order = [
        "S_neg", "CC2", "S_pos",
        "CC1",
        "DC_neg", "DC_pos",
        "A_neg", "PE", "A_pos",
    ]

    labels = [k for k in preferred_order if k in image_points_dict and k in model_points_dict]

    if len(labels) < min_points:
        raise ValueError(f"可用对应点不足，当前只有 {len(labels)} 个，至少需要 {min_points} 个。")

    object_points = np.array(
        [model_points_dict[k] for k in labels],
        dtype=np.float64
    )  # (N, 3), 单位 mm

    image_points = np.array(
        [image_points_dict[k] for k in labels],
        dtype=np.float64
    )  # (N, 2), 单位 pixel

    return object_points, image_points, labels


# =========================
# 6. rvec/tvec -> 齐次矩阵
# 输出 {}^C T_T
# =========================
def rt_to_homogeneous(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


# =========================
# 7. PnP 求解
# 先用 IPPE（平面点集更合适）
# 如果失败，再回退 ITERATIVE
# =========================
def solve_target_pose(
    image_points_dict: Dict[str, Tuple[float, float]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    model_csv_path: str | Path = MODEL_CSV,
):
    model_points_dict = load_model_points_from_csv(model_csv_path)

    object_points, image_points, labels = build_correspondences(
        image_points_dict=image_points_dict,
        model_points_dict=model_points_dict,
        min_points=4
    )

    print("=== labels ===")
    print(labels)
    print("\n=== object_points (mm) ===")
    print(object_points)
    print("\n=== image_points (px) ===")
    print(image_points)
    print("\n=== camera_matrix ===")
    print(camera_matrix)
    print("\n=== dist_coeffs ===")
    print(dist_coeffs.ravel())

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )

    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    if not ok:
        raise RuntimeError("solvePnP 求解失败。请检查二维点、相机内参与标签对应关系。")

    T_ct = rt_to_homogeneous(rvec, tvec)

    return {
        "labels": labels,
        "object_points": object_points,
        "image_points": image_points,
        "rvec": rvec,
        "tvec": tvec,
        "T_ct": T_ct,
    }


# =========================
# 8. 重投影误差
# =========================
def compute_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - image_points, axis=1)
    return float(np.mean(err))


# =========================
# 9. 可视化重投影
# 绿色: 实际二维点
# 红色: PnP重投影点
# 蓝线: 偏差
# =========================
def draw_reprojection(
    image: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    labels: List[str],
) -> np.ndarray:
    vis = image.copy()

    proj, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2)

    for (u, v), (pu, pv), name in zip(image_points, proj, labels):
        u, v = int(round(u)), int(round(v))
        pu, pv = int(round(pu)), int(round(pv))

        cv2.circle(vis, (u, v), 4, (0, 255, 0), -1)
        cv2.circle(vis, (pu, pv), 4, (0, 0, 255), -1)
        cv2.line(vis, (u, v), (pu, pv), (255, 0, 0), 1)
        cv2.putText(
            vis,
            name,
            (u + 4, v - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    return vis


# =========================
# 10. 示例主程序
# 注意：
# 这里已经自动读 latest_fitted_centers.json
# 但 camera_matrix 还需要你替换成真实标定值
# =========================
if __name__ == "__main__":

    image_points_dict = load_image_points_from_fitted_json(FITTED_JSON)

    print("=== image_points_dict ===")
    print(image_points_dict)

    # 这里替换成你的 Kinect 标定结果
    fx, fy = 1000.0, 1000.0
    cx, cy = 320.0, 240.0

    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)

    # 没有标定结果前先临时置零
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    result = solve_target_pose(
        image_points_dict=image_points_dict,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        model_csv_path=MODEL_CSV,
    )

    mean_err = compute_reprojection_error(
        result["object_points"],
        result["image_points"],
        result["rvec"],
        result["tvec"],
        camera_matrix,
        dist_coeffs
    )

    print("\n=== rvec ===")
    print(result["rvec"])

    print("\n=== tvec (mm) ===")
    print(result["tvec"])

    print("\n=== T_ct ===")
    print(result["T_ct"])

    print("\n=== mean reprojection error (pixel) ===")
    print(mean_err)
