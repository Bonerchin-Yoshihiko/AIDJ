"""音の伸び縮み（テンポ変更）。rubberband（高音質なテンポ調整ツール）があれば使い、なければ librosa で代用する。"""
import os
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf

SR = 44100


def rubberband(y, args):
    """rubberband に args を渡して y を変換する。見つからない・失敗したときは None。"""
    exe = shutil.which("rubberband")
    if not exe:
        return None
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # Windows で黒い窓が一瞬出ないように
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        sf.write(src, y.T, SR, subtype="FLOAT")
        args = [a.replace("{dir}", d) for a in args]
        for engine in (["--fine"], []):  # --fine は v3 以降。古い版なら付けずにやり直す
            try:
                subprocess.run([exe, "-q", *engine, *args, src, dst], check=True, capture_output=True, creationflags=flags)
                return sf.read(dst, dtype="float32", always_2d=True)[0].T
            except (OSError, subprocess.CalledProcessError):
                continue
    return None


def stretch(y, speed):
    """テンポを speed 倍にする（1.1 なら 10% 速く）。"""
    if abs(speed - 1) < 1e-4:
        return y
    out = rubberband(y, ["--tempo", str(speed)])
    if out is None:
        out = librosa.effects.time_stretch(y, rate=speed)  # 音質は少し落ちる
    return out


def warp(y, src_marks, dst_marks):
    """y の src_marks[k] 秒の位置が dst_marks[k] 秒に来るように、場所ごとに違う割合で伸び縮みさせる。

    テンポが少しずつ変わるブリッジに、元の曲の音を合わせるために使う。長さは dst_marks[-1] 秒になる。
    """
    length = int(round(dst_marks[-1] * SR))
    # rubberband は最初の伸び縮みの割合が 1 から離れていると、出力の頭を削りすぎて全体が前にずれる。
    # 前後に無音を足して、そこは 1:1 で始まる・終わるようにしてから、足した分を切り落とす。
    pad = SR
    src = np.pad(y[:, :int(round(src_marks[-1] * SR))], ((0, 0), (pad, pad)))
    marks = [(0, 0)] + [(int(round(s * SR)) + pad, int(round(t * SR)) + pad) for s, t in zip(src_marks, dst_marks)]
    marks.append((src.shape[1], marks[-1][1] + pad))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.writelines(f"{s} {t}\n" for s, t in marks)
        timemap = f.name
    try:
        out = rubberband(src, ["--timemap", timemap, "-D", f"{marks[-1][1] / SR:.6f}"])
    finally:
        os.remove(timemap)
    if out is not None:
        out = out[:, pad:pad + length]
    if out is None:  # rubberband がないときは、区間ごとに一定の割合で伸び縮みさせてつなぐ
        pieces = []
        for (s0, s1), (t0, t1) in zip(zip(src_marks, src_marks[1:]), zip(dst_marks, dst_marks[1:])):
            seg = y[:, int(s0 * SR):int(s1 * SR)]
            n = int(round(t1 * SR)) - int(round(t0 * SR))
            seg = librosa.effects.time_stretch(seg, rate=max(1e-3, seg.shape[1] / max(1, n))) if seg.shape[1] > 2048 else seg
            pieces.append(librosa.util.fix_length(seg, size=n, axis=-1))
        out = np.concatenate(pieces, axis=1)
    return librosa.util.fix_length(out, size=length, axis=-1)
