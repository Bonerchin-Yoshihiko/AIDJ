#!/usr/bin/env python3
"""CI 用: AI のパート分けがどれくらい正確かを、デモ曲の「正解」のパートと比べて測る。

SDR（信号対歪み比、dB）で表す。大きいほど正確で、0 dB 以下はほとんど分けられていない。
"""
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import separate  # noqa: E402


def sdr(ref, est):
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-12))


def main():
    assert separate.available(), "AI のパート分けが使えない（PyTorch・Demucs・モデルのどれかがない）"
    mix, sr = sf.read("demo/house_128.wav", dtype="float32", always_2d=True)
    section = slice(30 * sr, 50 * sr)  # 声の入っているサビ 20秒
    t = time.time()
    stems = separate.separate(mix[section].T)
    print(f"20秒ぶんのパート分け: {time.time() - t:.1f}秒（モデルの読み込み込み）")
    scores = {}
    for part in separate.PARTS:
        ref = sf.read(f"demo/stems/house_128_{part}.wav", dtype="float32", always_2d=True)[0][section].T
        scores[part] = sdr(ref, stems[part])
        print(f"  {separate.PART_NAMES[part]:4s}: SDR {scores[part]:5.1f} dB")
    assert scores["drums"] > 3 and scores["bass"] > 0, "パート分けの精度が低すぎる（モデルが正しく読み込めていない？）"
    print("パート分け OK")


if __name__ == "__main__":
    main()
