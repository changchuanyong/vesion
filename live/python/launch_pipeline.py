import subprocess
import time
import signal
import os
import sys
from pathlib import Path

# ===== 你改这里 =====
WORK_DIR = Path(__file__).resolve().parents[2]
CPP_WORK_DIR = WORK_DIR / "live" / "cpp"
PY_WORK_DIR = WORK_DIR / "live" / "python"

KINECT_EXE = CPP_WORK_DIR / "kinect_live_capture_atomic.exe"
ROI_WATCH_PY = PY_WORK_DIR / "roi.py"

PYTHON_EXE = WORK_DIR / ".venv" / "Scripts" / "python.exe"
START_DELAY = 2.0   # 先启动采图，再等 2 秒启动检测
# ===================


def start_process(cmd, cwd=None, name="process"):
    print(f"[START] {name}: {' '.join(map(str, cmd))}")
    return subprocess.Popen(cmd, cwd=cwd)


def main():
    processes = []
    python_cmd = str(PYTHON_EXE) if PYTHON_EXE.exists() else sys.executable

    try:
        # 1. 启动 Kinect 实时采图
        p1 = start_process(
            [str(KINECT_EXE)],
            cwd=str(WORK_DIR),
            name="kinect_capture"
        )
        processes.append(("kinect_capture", p1))

        # 等一会，让 latest.jpg 先开始生成
        time.sleep(START_DELAY)

        # 2. 启动 ROI 检测与裁剪
        p2 = start_process(
            [python_cmd, str(ROI_WATCH_PY)],
            cwd=str(WORK_DIR),
            name="roi_watch"
        )
        processes.append(("roi_watch", p2))

        print("\nAll processes started.")
        print("Press Ctrl+C to stop all.\n")

        while True:
            for name, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    print(f"[EXIT] {name} exited with code {ret}")
                    raise RuntimeError(f"{name} stopped unexpectedly.")
            time.sleep(0.5)

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
