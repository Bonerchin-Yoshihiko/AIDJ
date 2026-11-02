#!/usr/bin/env python3
"""配布用のアプリを作る。Windows なら dist/AI DJ/AI DJ.exe、Mac なら dist/AI DJ.app ができる。

  pip install -r requirements.txt pyinstaller
  python build.py

PyInstaller は動かしている OS 向けのアプリしか作れないので、
Windows 版は Windows で、Mac 版は Mac で実行する。
"""
import os

import PyInstaller.__main__

PyInstaller.__main__.run([
    "app.py",
    "--name=AI DJ",
    "--windowed",          # コンソール画面を出さない（Mac では .app になる）
    "--noconfirm",
    "--clean",
    f"--add-data=web{os.pathsep}web",
    "--collect-data=librosa",
])
