"""曲をパート（ドラム・ベース・ボーカル・その他）に分ける。

Meta が公開している AI「Demucs」（htdemucs モデル）を使う。モデルの重みは models/htdemucs.th
（配布用アプリには同梱）。PyTorch の読み込みは数秒かかるので、実際に分けるときまで import しない。
"""
import hashlib
import importlib.util
import os
import sys
import threading

import numpy as np

PARTS = ("drums", "bass", "vocals", "other")
PART_NAMES = {"drums": "ドラム", "bass": "ベース", "vocals": "ボーカル", "other": "その他"}
HERE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

_model = None
_lock = threading.Lock()
_cache = {}


def model_path():
    return os.environ.get("AIDJ_MODEL") or os.path.join(HERE, "models", "htdemucs.th")


def available():
    """AI のパート分けが使えるか（PyTorch・Demucs・モデルの重みがそろっているか）。"""
    try:
        return os.path.exists(model_path()) and all(importlib.util.find_spec(m) for m in ("torch", "demucs"))
    except (ImportError, ValueError):
        return False


def _load():
    global _model
    if _model is None:
        import torch
        from demucs.states import load_model
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        _model = load_model(model_path())
        _model.eval()
    return _model


def separate(y):
    """y: (2, n) の音（44.1kHz）を {"drums": (2, n), "bass": ..., "vocals": ..., "other": ...} に分ける。"""
    y = np.ascontiguousarray(y, dtype=np.float32)
    key = hashlib.sha1(y.tobytes()).hexdigest()
    with _lock:  # 同じ区間は何度も分けない（ミックスし直したときなど）
        if key in _cache:
            return _cache[key]
        import torch
        from demucs.apply import apply_model
        model = _load()
        wav = torch.from_numpy(y)
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            out = apply_model(model, ((wav - mean) / std)[None], shifts=0, split=True, overlap=0.25)[0]
        out = out * std + mean
        stems = {name: out[i].numpy() for i, name in enumerate(model.sources)}
        if len(_cache) > 16:
            _cache.pop(next(iter(_cache)))
        _cache[key] = stems
        return stems
