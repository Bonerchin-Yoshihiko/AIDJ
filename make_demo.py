#!/usr/bin/env python3
"""動作確認用のデモ曲を合成する（実際の曲がなくても試せるように）。

  demo/house_124.wav  124 BPM・Aマイナー。17〜32小節目が「サビ」
  demo/house_128.wav  128 BPM・Cメジャー。17〜32小節目が「サビ」
  demo/piano_free.wav テンポが揺れるピアノ（ビートなし）
  demo/groove_90.wav  90 BPM・F#メジャー（ほかの曲とテンポもキーも離れている）。17〜32小節目が「サビ」

house_128 のサビには声も入れている（ci/fixtures/voice.wav: 音声合成 espeak-ng で作った英語のフレーズ）。
demo/stems/ には、AI のパート分けがどれくらい正確かを確かめるための「正解」のパートを書き出す。
"""
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

SR = 44100
rng = np.random.default_rng(0)


def env(n, decay):
    return np.exp(-np.arange(n) / (decay * SR))


def kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    freq = 45 + 75 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * env(n, 0.12)


def noise_hit(decay, band):
    n = int(decay * 6 * SR)
    sos = butter(2, band if len(band) == 2 else band[0], btype="bandpass" if len(band) == 2 else "highpass", fs=SR, output="sos")
    return sosfilt(sos, rng.standard_normal(n)) * env(n, decay)


def tone(freq, dur, decay=None, harmonics=(1.0, 0.5, 0.25)):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = sum(a * np.sin(2 * np.pi * freq * (i + 1) * t) for i, a in enumerate(harmonics))
    y *= env(n, decay) if decay else np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.05)
    return y


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


def add(buf, x, at):
    i = int(at * SR)
    j = min(len(buf), i + len(x))
    if i < len(buf):
        buf[i:j] += x[: j - i]


def voice_phrases():
    """ci/fixtures/voice.wav の4つのフレーズを 44.1kHz で返す。"""
    here = os.path.dirname(os.path.abspath(__file__))
    y, sr = sf.read(os.path.join(here, "ci", "fixtures", "voice.wav"))
    y = resample_poly(y, SR // sr, 1)
    lengths = [int(n) * (SR // sr) for n in open(os.path.join(here, "ci", "fixtures", "voice_phrases.txt")).read().split()]
    gap, phrases, pos = int(0.25 * SR), [], 0
    for n in lengths:
        phrases.append(y[pos:pos + n])
        pos += n + gap
    return phrases


def house(bpm, root, chords, lead, bars=48, chorus=(16, 32), vocal=None):
    """パートごと（ドラム・ベース・その他・声）の音を返す。曲はこれを足したもの。"""
    beat = 60 / bpm
    n = int((bars * 4 + 4) * beat * SR)
    parts = {name: np.zeros(n) for name in ("drums", "bass", "other", "vocals")}
    k, clap, hat = kick(), noise_hit(0.05, [900, 3000]), noise_hit(0.015, [7000])
    for bar in range(bars):
        in_chorus = chorus[0] <= bar < chorus[1]
        ch = chords[(bar // 2) % len(chords)]
        for b in range(4):
            t = (bar * 4 + b) * beat
            add(parts["drums"], 0.9 * k, t)
            add(parts["drums"], 0.25 * hat, t + beat / 2)
            if b in (1, 3):
                add(parts["drums"], 0.35 * clap, t)
            add(parts["bass"], 0.35 * tone(midi(root + ch[0] - 12), beat * 0.45, harmonics=(1, 0.6, 0.3, 0.15)), t + beat / 2)
        if bar >= 8:  # 9小節目からコード
            for m in ch:
                add(parts["other"], 0.08 * tone(midi(root + m + 12), beat * 4, harmonics=(1, 0.3)), bar * 4 * beat)
        if in_chorus:  # サビだけリード
            for i, m in enumerate(lead[(bar % 2) * 4:(bar % 2) * 4 + 4]):
                add(parts["other"], 0.12 * tone(midi(root + m + 24), beat * 0.9, decay=0.3), (bar * 4 + i) * beat)
            if vocal and bar % 2 == 0:  # サビは2小節ごとに声のフレーズ
                add(parts["vocals"], 1.0 * vocal[(bar // 2) % len(vocal)], bar * 4 * beat)
    return parts


def free_piano(seconds=70):
    out = np.zeros(int(seconds * SR))
    t, i = 0.5, 0
    notes = [57, 60, 64, 67, 65, 64, 62, 60, 59, 62, 65, 69]
    while t < seconds - 3:
        bpm = 72 + 22 * np.sin(t / 9)  # テンポがゆっくり揺れる
        add(out, 0.25 * tone(midi(notes[i % len(notes)]), 3, decay=0.8, harmonics=(1, 0.4, 0.2, 0.1)), t)
        if i % 4 == 0:
            add(out, 0.2 * tone(midi(notes[i % len(notes)] - 24), 4, decay=1.2), t)
        t += 60 / bpm * rng.choice([1, 1, 0.5, 1.5])
        i += 1
    return out


def save(name, mono, parts=None):
    stereo = lambda x: np.stack([x, np.roll(x, 40)], axis=1)  # ほんの少しだけ左右差
    scale = 1 / (np.max(np.abs(stereo(mono))) * 1.1)
    sf.write(os.path.join("demo", name), stereo(mono) * scale, SR)
    print(f"demo/{name}  {len(mono) / SR:.1f}秒")
    for part, x in (parts or {}).items():
        os.makedirs(os.path.join("demo", "stems"), exist_ok=True)
        sf.write(os.path.join("demo", "stems", name.replace(".wav", f"_{part}.wav")), stereo(x) * scale, SR)


def save_parts(name, parts):
    save(name, sum(parts.values()), {k: v for k, v in parts.items() if v.any()})


if __name__ == "__main__":
    os.makedirs("demo", exist_ok=True)
    save_parts("house_124.wav", house(124, 57, [[0, 3, 7], [-4, 0, 3], [-2, 2, 5], [-5, -2, 2]], [0, 3, 7, 10, 12, 10, 7, 3]))
    save_parts("house_128.wav", house(128, 60, [[0, 4, 7], [-3, 0, 4], [-7, -3, 0], [-5, -1, 2]], [12, 11, 7, 4, 5, 7, 11, 12],
                                      vocal=voice_phrases()))
    save("piano_free.wav", free_piano())
    save_parts("groove_90.wav", house(90, 54, [[0, 4, 7], [5, 9, 12], [-3, 0, 4], [7, 11, 14]], [7, 9, 11, 12, 11, 9, 7, 4], bars=40))
