import os
import subprocess
import sys
import time
from pathlib import Path

# ===== 你改这里 =====
WORK_DIR = Path(__file__).resolve().parents[2]
CPP_WORK_DIR = WORK_DIR / "live" / "cpp"
PY_WORK_DIR = WORK_DIR / "live" / "python"
LIVE_DIR = WORK_DIR / "dataset" / "live"

KINECT_EXE = CPP_WORK_DIR / "kinect_live_capture_atomic.exe"
ROI_WATCH_PY = PY_WORK_DIR / "roi.py"
CANNY_WATCH_PY = PY_WORK_DIR / "Canny.py"
ENHANCE_PY = PY_WORK_DIR / "enhance.py"
CONTOURS_PY = PY_WORK_DIR / "latest_candidate_contours.py"
PNP_STEP_PY = PY_WORK_DIR / "pnp2.py"

PYTHON_EXE = WORK_DIR / ".yolo_env" / "Scripts" / "python.exe"
START_DELAY = 2.0
ALLOW_NO_CAMERA_DURING_DEBUG = True
KEY_WINDOWS_ONLY = True

ROI_FILE = LIVE_DIR / "latest_roi.jpg"
ENHANCED_FILE = LIVE_DIR / "latest_roi_enhanced.jpg"
EDGES_FILE = LIVE_DIR / "latest_edges.jpg"

POST_ROI_COOLDOWN = 0.05
EDGE_WAIT_TIMEOUT = 3.0
POLL_INTERVAL = 0.05
# ===================


def start_process(cmd, cwd=None, name="process", env=None):
    print(f"[START] {name}: {' '.join(map(str, cmd))}")
    return subprocess.Popen(cmd, cwd=cwd, env=env)


def run_once(cmd, cwd=None, name="step", env=None) -> bool:
    print(f"[RUN ] {name}: {' '.join(map(str, cmd))}")
    completed = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        print(f"[FAIL] {name} exited with code {completed.returncode}")
        return False
    print(f"[ OK ] {name}")
    return True


def wait_for_newer_file(path: Path, min_mtime: float, timeout_s: float, poll_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if path.exists() and path.stat().st_mtime > min_mtime:
                return True
        except OSError:
            pass
        time.sleep(poll_s)
    return False


def main():
    processes = []
    python_cmd = str(PYTHON_EXE) if PYTHON_EXE.exists() else sys.executable

    child_env = os.environ.copy()
    child_env["VISION_PIPELINE_MODE"] = "1"
    if KEY_WINDOWS_ONLY:
        child_env["VISION_KEY_WINDOWS"] = "1"

    if not PNP_STEP_PY.exists():
        raise FileNotFoundError(f"PnP step script not found: {PNP_STEP_PY}")

    try:
        p1 = start_process(
            [str(KINECT_EXE)],
            cwd=str(WORK_DIR),
            name="kinect_capture",
            env=child_env,
        )
        processes.append(("kinect_capture", p1))

        time.sleep(START_DELAY)

        p2 = start_process(
            [python_cmd, str(ROI_WATCH_PY)],
            cwd=str(WORK_DIR),
            name="roi_watch",
            env=child_env,
        )
        processes.append(("roi_watch", p2))

        p3 = start_process(
            [python_cmd, str(CANNY_WATCH_PY)],
            cwd=str(WORK_DIR),
            name="canny_watch",
            env=child_env,
        )
        processes.append(("canny_watch", p3))

        print("\nAll processes started.")
        print("Press Ctrl+C to stop all.\n")
        print(f"PnP step script: {PNP_STEP_PY.name}")
        print("Pipeline order: roi -> enhance -> canny -> latest_candidate_contours -> pnp")

        last_roi_mtime = 0.0

        while True:
            for name, proc in list(processes):
                ret = proc.poll()
                if ret is not None:
                    if name == "kinect_capture" and ALLOW_NO_CAMERA_DURING_DEBUG:
                        print(
                            f"[WARN] {name} exited with code {ret}. "
                            "Camera may be disconnected; continuing in debug mode."
                        )
                        processes = [(n, p) for n, p in processes if n != name]
                        continue

                    print(f"[EXIT] {name} exited with code {ret}")
                    raise RuntimeError(f"{name} stopped unexpectedly.")

            try:
                roi_mtime = ROI_FILE.stat().st_mtime if ROI_FILE.exists() else 0.0
            except OSError:
                roi_mtime = 0.0

            if roi_mtime <= 0.0 or roi_mtime == last_roi_mtime:
                time.sleep(0.5)
                continue

            last_roi_mtime = roi_mtime
            print(f"\n[TRIG] New ROI detected @ {time.strftime('%H:%M:%S')}")

            if not run_once([python_cmd, str(ENHANCE_PY)], cwd=str(WORK_DIR), name="enhance", env=child_env):
                continue

            try:
                enhanced_mtime = ENHANCED_FILE.stat().st_mtime if ENHANCED_FILE.exists() else roi_mtime
            except OSError:
                enhanced_mtime = roi_mtime

            edge_ok = wait_for_newer_file(
                path=EDGES_FILE,
                min_mtime=enhanced_mtime - POST_ROI_COOLDOWN,
                timeout_s=EDGE_WAIT_TIMEOUT,
                poll_s=POLL_INTERVAL,
            )
            if not edge_ok:
                print("[WARN] latest_edges.jpg was not updated in time; continue with current edge file.")

            if not run_once(
                [python_cmd, str(CONTOURS_PY)],
                cwd=str(WORK_DIR),
                name="latest_candidate_contours",
                env=child_env,
            ):
                continue

            run_once([python_cmd, str(PNP_STEP_PY)], cwd=str(WORK_DIR), name=PNP_STEP_PY.stem, env=child_env)

    except KeyboardInterrupt:
        print("\nStopping all processes...")
    except Exception as e:
        print(f"\nLauncher error: {e}")
    finally:
        for name, proc in processes:
            if proc.poll() is None:
                print(f"[STOP] {name}")
                proc.terminate()

        time.sleep(1.0)

        for name, proc in processes:
            if proc.poll() is None:
                print(f"[KILL] {name}")
                proc.kill()

        print("All stopped.")


if __name__ == "__main__":
    main()
