#!/usr/bin/env python3
"""CI 用: exe をアプリのウィンドウで起動し、画面が表示されるか確かめてスクリーンショットを撮る。

  python ci/screenshot.py "dist/AI DJ/AI DJ.exe"
"""
import os
import subprocess
import sys
import tempfile
import time

from PIL import ImageGrab

LOG = os.path.join(tempfile.gettempdir(), "aidj.log")
OUT = "ci-output"


def main():
    os.makedirs(OUT, exist_ok=True)
    start = os.path.getsize(LOG) if os.path.exists(LOG) else 0
    proc = subprocess.Popen([sys.argv[1], "--port", "8766"])
    try:
        # 画面の JavaScript が動けば、波形ライブラリの読み込みがログに残る
        for _ in range(90):
            time.sleep(1)
            if os.path.exists(LOG):
                with open(LOG, encoding="utf-8", errors="replace") as f:
                    f.seek(start)
                    if "regions.esm.js" in f.read():
                        break
        else:
            sys.exit("ウィンドウに画面が読み込まれませんでした")
        time.sleep(3)
        ImageGrab.grab().save(os.path.join(OUT, "window.png"))
        print("ウィンドウ表示 OK（ci-output/window.png）")
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
