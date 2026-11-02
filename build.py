#!/usr/bin/env python3
"""配布用のアプリを作る。Windows なら dist/AI DJ/AI DJ.exe、Mac なら dist/AI DJ.app ができる。

  pip install -r requirements.txt pyinstaller
  pip install -r requirements-ai.txt && pip install --no-deps demucs==4.1.0   # AI のパート分けも入れるとき
  （models/htdemucs.th にモデルを置く。README を参照）
  python build.py

PyInstaller は動かしている OS 向けのアプリしか作れないので、
Windows 版は Windows で、Mac 版は Mac で実行する。
"""
import os

import PyInstaller.__main__

args = [
    "app.py",
    "--name=AI DJ",
    "--windowed",          # コンソール画面を出さない（Mac では .app になる）
    "--noconfirm",
    "--clean",
    f"--add-data=web{os.pathsep}web",
    "--collect-data=librosa",
]
if os.path.exists(os.path.join("models", "htdemucs.th")):  # AI のパート分け（モデルと Demucs）も入れる
    args.append(f"--add-data=models{os.pathsep}models")
    # モデルのファイルからクラスを読み込むので、使う部品を明示しておく
    args += [f"--hidden-import=demucs.{m}" for m in ("apply", "states", "htdemucs", "hdemucs", "demucs", "transformer", "spec", "wiener", "utils")]
PyInstaller.__main__.run(args)
