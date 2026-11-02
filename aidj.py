#!/usr/bin/env python3
"""AI DJ 試作: 曲ごとに「聴かせたい範囲」を指定して、テンポを揃えてつなぐ。

  python aidj.py "曲A.mp3@1:05-2:10" "曲B.mp3@0:45-1:50x2" -o mix.wav

範囲の書き方:
  曲.mp3@1:05-2:10    1分5秒〜2分10秒
  曲.mp3@65-130       秒数でもOK
  曲.mp3@1:05-        1分5秒から最後まで（@-2:10 なら頭から2分10秒まで）
  曲.mp3@1:05-1:20x4  範囲を4回ループ
  曲.mp3              範囲なし = 曲全体

指定した範囲は必ずそのまま流れる。つなぎ（クロスフェード）は範囲の外側、
つまり前の曲の範囲の「後ろ」と次の曲の範囲の「前」を重ねて作る。
"""
import argparse
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SR = 44100
STEADY_JITTER = 0.03  # 拍の位置のばらつき（秒）がこれ以下なら「テンポが一定」とみなす
LOW_SOS = butter(4, 180, btype="lowpass", fs=SR, output="sos")  # ベーススワップの境目
ECHO_SOS = butter(2, 300, btype="highpass", fs=SR, output="sos")  # エコーに低音を残さない
SPEC_RE = re.compile(r"^(?P<path>.+)@(?P<start>[\d:.]*)-(?P<end>[\d:.]*)(?:x(?P<loops>\d+))?$")


class MixError(Exception):
    """ユーザーに見せるエラー（範囲が短すぎる、など）。"""


@dataclass
class Spec:
    path: str
    start: float | None
    end: float | None
    loops: int = 1
    name: str | None = None  # 表示名（省略時はファイル名）

    @property
    def label(self):
        return self.name or os.path.basename(self.path)


@dataclass
class Track:
    spec: Spec
    y: np.ndarray          # (2, n) 元の音声
    start: float           # 拍に合わせ直した範囲（秒）
    end: float
    period: float | None   # 1拍の長さ（秒）。None = テンポが揺れていてビートマッチできない


@dataclass
class Clip:
    audio: np.ndarray      # (2, n) テンポ調整・ループ済みの音声
    rs: int                # 範囲の開始サンプル
    re: int                # 範囲の終了サンプル
    beat: int              # この曲の1拍のサンプル数（テンポ調整後）
    matched: bool          # ミックスのテンポに揃っているか
    gain: float = 1.0


def parse_time(s):
    if not s:
        return None
    sec = 0.0
    for part in s.split(":"):
        sec = sec * 60 + float(part)
    return sec


def parse_spec(arg):
    m = SPEC_RE.match(arg)
    if not m:
        return Spec(arg, None, None)
    return Spec(m["path"], parse_time(m["start"]), parse_time(m["end"]), int(m["loops"] or 1))


def fmt(sec):
    sec = round(sec, 2)
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def load(path):
    """音声を読み込んで (2, n) の配列にする。ふつうの方法で読めなければ WAV を直接読んでみる。"""
    try:
        y, _ = librosa.load(path, sr=SR, mono=False)
    except Exception as e:
        y = read_wav_lenient(path)
        if y is None:
            raise MixError(f"{os.path.basename(path)}: {why_unreadable(path)}") from e
    if y.size == 0:  # ヘッダーに「データの大きさ 0」と書かれた WAV は、ふつうに読むと空になる
        y = read_wav_lenient(path)
        if y is None or y.size == 0:
            raise MixError(f"{os.path.basename(path)}: 音が入っていないか、ファイルが壊れています")
    if y.ndim == 1:
        y = np.stack([y, y])
    return y[:2]


WAV_TAGS = {1: "pcm", 3: "float"}


def read_wav_lenient(path):
    """ヘッダーが壊れた WAV を、中身を直接見て読む（録音ソフトが途中で落ちたときなど）。

    データの大きさが 0 や実際より大きいと書かれていても、ファイルの最後まで読む。
    PCM（8/16/24/32bit）と浮動小数点（32/64bit）だけ。読めなければ None。
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        return None
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt " and size >= 16:
            tag, channels, sr, _, _, bits = struct.unpack("<HHIIHH", data[body:body + 16])
            if tag == 0xFFFE and size >= 26:  # WAVE_FORMAT_EXTENSIBLE: 本当の形式は後ろに書いてある
                tag = struct.unpack("<H", data[body + 24:body + 26])[0]
            fmt = (WAV_TAGS.get(tag), channels, sr, bits)
        elif cid == b"data" and fmt:
            kind, channels, sr, bits = fmt
            end = len(data) if size in (0, 0xFFFFFFFF) or body + size > len(data) else body + size
            raw = data[body:end]
            raw = raw[:len(raw) - len(raw) % (channels * bits // 8)]
            if kind == "pcm" and bits == 24:
                b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
                x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8) / 2 ** 23
            elif kind == "pcm" and bits in (8, 16, 32):
                dtype = {8: np.uint8, 16: np.int16, 32: np.int32}[bits]
                x = np.frombuffer(raw, dtype).astype(np.float64)
                x = (x - 128) / 128 if bits == 8 else x / 2 ** (bits - 1)
            elif kind == "float" and bits in (32, 64):
                x = np.frombuffer(raw, np.float32 if bits == 32 else np.float64)
            else:
                return None
            y = x.reshape(-1, channels).T.astype(np.float32)
            return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        if size == 0xFFFFFFFF:
            break
        pos = body + size + (size & 1)
    return None


def why_unreadable(path):
    """読めなかった理由を、ファイルの先頭を見て説明する。"""
    with open(path, "rb") as f:
        head = f.read(64)
    if head[4:8] == b"ftyp":
        return "中身は m4a / AAC 形式です（拡張子だけ変わっています）。mp3 か wav に変換してください"
    if head[:4] == b"\x30\x26\xb2\x75":
        return "中身は WMA 形式です。mp3 か wav に変換してください"
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return "この WAV は特殊な圧縮形式で保存されていて読めません。ふつうの WAV（PCM）か mp3 に変換してください"
    return "音声ファイルとして読めませんでした（壊れているか、対応していない形式です。mp3 / wav / flac / ogg が使えます）"


_beat_cache = {}


def beat_times(path, mono):
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _beat_cache:
        _, _beat_cache[key] = librosa.beat.beat_track(y=mono, sr=SR, units="time")
    return _beat_cache[key]


def peaks_of(mono, n=3000):
    """波形表示用に、音量の山を n 個に間引く（0〜1）。"""
    hop = max(1, len(mono) // n)
    peaks = np.abs(mono[:hop * (len(mono) // hop)]).reshape(-1, hop).max(axis=1)
    return np.round(peaks / (peaks.max() + 1e-9), 3).tolist()


def suggest_range(mono, beats, bars=16):
    """一番盛り上がっている（音量が大きい）bars 小節を、おすすめ範囲として返す。"""
    dur = len(mono) / SR
    period = float(np.median(np.diff(beats))) if len(beats) > 8 else 0.5
    length = min(bars * 4 * period, dur * 0.6)
    hop = 2048
    rms = librosa.feature.rms(y=mono, frame_length=4096, hop_length=hop)[0]
    win = max(1, int(length * SR / hop))
    if len(rms) <= win:
        return 0.0, dur
    score = np.convolve(rms, np.ones(win) / win, mode="valid")
    start = np.argmax(score) * hop / SR
    if len(beats):
        start = float(beats[np.argmin(np.abs(beats - start))])
    return start, min(dur, start + length)


def overview(path, y=None):
    """アプリで曲を読み込んだときの情報（長さ・波形・おおよそのBPM・おすすめ範囲）。"""
    y = load(path) if y is None else y
    mono = librosa.to_mono(y)
    beats = beat_times(path, mono)
    grid = fit_grid(beats, 0, len(mono) / SR)
    bpm = 60 / grid[1] if grid else None
    start, end = suggest_range(mono, beats)
    return {
        "duration": len(mono) / SR,
        "peaks": peaks_of(mono),
        "bpm": round(bpm, 1) if bpm else None,
        "steady": grid is not None and grid[2] <= STEADY_JITTER,
        "suggested": [start, end],
    }


def fit_grid(beats, lo, hi):
    """lo〜hi 秒の拍を等間隔の格子 t = t0 + k * period に当てはめる。"""
    b = beats[(beats >= lo) & (beats <= hi)]
    if len(b) < 8:
        return None
    p0 = np.median(np.diff(b))
    b = b[np.concatenate([[True], np.diff(b) > 0.5 * p0])]  # 二重に拾った拍を除く
    k = np.concatenate([[0], np.cumsum(np.round(np.diff(b) / p0))])
    period, t0 = np.polyfit(k, b, 1)
    return t0, period, float(np.std(b - (t0 + k * period)))


def kick_phase(y, t0, period, lo, hi):
    """拍の位置をキック（低音のアタック）が一番そろう位置にずらす。

    ビート検出はハイハットや裏拍のベースにつられて半拍ずれることがあるので、
    低音域のアタックだけを見て「表拍」を決め直す。
    """
    hop = 256
    S = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=hop, n_mels=8, fmax=160)
    env = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=SR, hop_length=hop)
    ks = np.arange(math.ceil((max(lo, 0) - t0) / period), math.floor((hi - t0) / period))
    phis = np.arange(0, period, hop / SR)
    idx = lambda phi: np.clip(np.round((t0 + phi + ks * period) * SR / hop).astype(int), 0, len(env) - 1)
    best = phis[int(np.argmax([env[idx(phi)].sum() for phi in phis]))]
    return t0 + best - 0.01  # アタックの少し手前を拍の頭にする（切ったときに頭が欠けないように）


def analyze(spec, margin):
    y = load(spec.path)
    dur = y.shape[1] / SR
    start = 0.0 if spec.start is None else min(spec.start, dur)
    end = dur if spec.end is None else min(spec.end, dur)
    if end - start < 1:
        raise MixError(f"範囲が短すぎます: {spec.label} ({fmt(start)}-{fmt(end)})")

    mono = librosa.to_mono(y)
    beats = beat_times(spec.path, mono)
    grid = fit_grid(beats, start - margin, end + margin)
    if grid is None or grid[2] > STEADY_JITTER:
        return Track(spec, y, start, end, None)

    # 範囲の端を一番近い拍に寄せる（つなぎ目でリズムがずれないように）
    t0, period, _ = grid
    t0 = kick_phase(mono, t0, period, start - margin, end + margin)
    s = t0 + round((start - t0) / period) * period
    e = t0 + round((end - t0) / period) * period
    s = max(0.0, s)
    e = e - period if e > dur else e
    if e - s < period:
        raise MixError(f"範囲が短すぎます: {spec.label}")
    return Track(spec, y, s, e, period)


def fold(native, bpm):
    """倍・半分のテンポも同じ仲間とみなし、bpm に一番近い形にする（70 BPM と 140 BPM など）。"""
    return native * 2.0 ** max(-1, min(1, round(math.log2(bpm / native))))


def choose_tempo(tracks, bpm, max_stretch):
    """ミックス全体のテンポを決める。なるべく多くの曲を、なるべく小さい変化で揃えられる値を選ぶ。"""
    steady = [60 / t.period for t in tracks if t.period]
    if bpm or not steady:
        return bpm

    def score(cand):
        ok = [d for d in (abs(cand / fold(b, cand) - 1) for b in steady) if d <= max_stretch]
        return len(ok), -max(ok, default=0)

    folded = [fold(b, steady[0]) for b in steady]
    return max(folded + [float(np.median(folded))], key=score)


def rubberband(y, speed):
    """rubberband（高音質なテンポ調整ツール）で伸び縮みさせる。見つからなければ None。"""
    exe = shutil.which("rubberband")
    if not exe:
        return None
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # Windows で黒い窓が一瞬出ないように
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
        sf.write(src, y.T, SR, subtype="FLOAT")
        for engine in (["--fine"], []):  # --fine は v3 以降。古い版なら付けずにやり直す
            try:
                subprocess.run([exe, "-q", *engine, "--tempo", str(speed), src, dst],
                               check=True, capture_output=True, creationflags=flags)
                return sf.read(dst, dtype="float32", always_2d=True)[0].T
            except (OSError, subprocess.CalledProcessError):
                continue
    return None


def stretch(y, speed):
    if abs(speed - 1) < 1e-4:
        return y
    out = rubberband(y, speed)
    if out is None:
        # rubberband がなければ librosa で代用（音質は少し落ちる）
        out = librosa.effects.time_stretch(y, rate=speed)
    return out


def loop(y, rs, re_, n):
    """範囲 [rs, re_) を n 回くり返す。後ろの余白はそのまま残す。"""
    body = y[:, rs:re_]
    tail = y[:, re_:re_ + int(0.005 * SR)]  # つなぎ目は本来の続きの音から 5ms でクロスフェード
    w = np.linspace(0, 1, tail.shape[1])
    seam = body.copy()
    seam[:, :tail.shape[1]] = body[:, :tail.shape[1]] * w + tail * (1 - w)
    out = np.concatenate([y[:, :rs], body] + [seam] * (n - 1) + [y[:, re_:]], axis=1)
    return out, rs + n * (re_ - rs)


def prepare(t, bpm, max_stretch, need):
    """テンポを揃え、範囲の前後に need 秒（ミックス上）の余白をつけて切り出す。"""
    speed, matched = 1.0, False
    if t.period and bpm:
        native = fold(60 / t.period, bpm)
        if abs(bpm / native - 1) <= max_stretch:
            speed, matched = bpm / native, True

    pad = need * speed + 0.5
    a = max(0, int((t.start - pad) * SR))
    b = min(t.y.shape[1], int((t.end + pad) * SR))
    y = stretch(t.y[:, a:b], speed)
    rs = int(round((t.start - a / SR) / speed * SR))
    re_ = min(y.shape[1], int(round((t.end - a / SR) / speed * SR)))
    if t.spec.loops > 1:
        y, re_ = loop(y, rs, re_, t.spec.loops)
    beat = int(round((t.period or 0.5) / speed * SR))
    return Clip(y, rs, re_, beat, matched), speed


def crossfade_gains(n, beat):
    """高音は等パワーで入れ替え、低音は真ん中でサッと入れ替える（ベーススワップ）。"""
    t = np.linspace(0, 1, n)
    ramp = max(1, min(beat // 2, n // 4))
    lo_in = np.clip((np.arange(n) - (n // 2 - ramp // 2)) / ramp, 0, 1)
    return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2), 1 - lo_in, lo_in


def apply_gains(x, hi, lo):
    low = sosfiltfilt(LOW_SOS, x, axis=1)
    return (x - low) * hi + low * lo


def echo_tail(c):
    """範囲の最後の1拍をエコーで残響させる（テンポが合わない曲同士のつなぎ用）。"""
    src = sosfiltfilt(ECHO_SOS, c.audio[:, max(0, c.re - c.beat):c.re] * c.gain, axis=1)
    tail = np.zeros((2, 4 * c.beat + src.shape[1]))
    for k, g in enumerate([0.5, 0.25, 0.12, 0.06]):
        tail[:, k * c.beat:k * c.beat + src.shape[1]] += g * src
    return tail


def short_fade(x, n, out):
    n = min(n, x.shape[1])
    w = np.linspace(0, 1, n)
    if out:
        x[:, -n:] *= w[::-1]
    else:
        x[:, :n] *= w


def build_mix(clips, beat, bars):
    """各曲を並べる。範囲内には手を加えず、範囲の外側の余白同士を重ねてつなぐ。"""
    want = bars * 4 * beat
    pre, post, kinds = [0] * len(clips), [0] * len(clips), []
    for i, (a, b) in enumerate(zip(clips, clips[1:])):
        n = 0
        if a.matched and b.matched:
            beats = min(a.audio.shape[1] - a.re, b.rs, want) // beat
            n = (beats - beats % 4 if beats >= 4 else beats) * beat  # できるだけ小節単位で
        post[i] = pre[i + 1] = n
        kinds.append(n // beat)
    post[-1] = min(clips[-1].audio.shape[1] - clips[-1].re, want)

    pieces, offset, starts = [], -clips[0].rs, []
    for i, c in enumerate(clips):
        lo, hi = c.rs - pre[i], c.re + post[i]
        x = c.audio[:, lo:hi] * c.gain
        if pre[i]:
            _, g_hi, _, g_lo = crossfade_gains(pre[i], beat)
            x[:, :pre[i]] = apply_gains(x[:, :pre[i]], g_hi, g_lo)
        else:
            short_fade(x, int(0.005 * SR), out=False)
        if post[i]:
            g_hi, _, g_lo, _ = crossfade_gains(post[i], beat)
            x[:, -post[i]:] = apply_gains(x[:, -post[i]:], g_hi, g_lo)
        else:
            short_fade(x, int(0.01 * SR), out=True)
            pieces.append((offset + c.re, echo_tail(c)))
        pieces.append((offset + lo, x))
        starts.append(((offset + c.rs) / SR, (offset + c.re) / SR))
        if i + 1 < len(clips):
            offset += c.re + post[i] - clips[i + 1].rs

    out = np.zeros((2, max(p + x.shape[1] for p, x in pieces)))
    for p, x in pieces:
        out[:, p:p + x.shape[1]] += x
    return out, starts, kinds


def make_mix(specs, output, bars=8, bpm=None, max_stretch=0.15, progress=None):
    """specs を順につないで output に書き出し、どう組んだかを dict で返す。"""
    report = progress or (lambda frac, msg: None)
    n = len(specs)
    tracks = []
    for i, s in enumerate(specs):
        report(0.4 * i / n, f"解析中 ({i + 1}/{n}) {s.label}")
        tracks.append(analyze(s, margin=bars * 4 * 0.6 + 2))
    bpm = choose_tempo(tracks, bpm, max_stretch)
    beat = int(round(60 / bpm * SR)) if bpm else int(0.5 * SR)
    need = bars * 4 * beat / SR

    clips, speeds = [], []
    for i, t in enumerate(tracks):
        report(0.4 + 0.45 * i / n, f"テンポ調整中 ({i + 1}/{n}) {t.spec.label}")
        c, sp = prepare(t, bpm, max_stretch, need)
        clips.append(c)
        speeds.append(sp)
    rms = [np.sqrt(np.mean(c.audio[:, c.rs:c.re] ** 2)) + 1e-9 for c in clips]
    for c, r in zip(clips, rms):  # 曲ごとの音量差をならす
        c.gain = float(np.clip(np.median(rms) / r, 0.25, 4))

    report(0.9, "つなぎを作成中")
    out, starts, kinds = build_mix(clips, beat, bars)
    peak = np.max(np.abs(out))
    if peak > 0.98:
        out *= 0.98 / peak
    report(0.95, "書き出し中")
    sf.write(output, out.T, SR)

    items = []
    for i, (t, c, sp, (ms, me)) in enumerate(zip(tracks, clips, speeds, starts)):
        items.append({
            "path": t.spec.path,
            "bpm": 60 / t.period if t.period else None,
            "stretch": (sp - 1) if c.matched else None,
            "start": t.start, "end": t.end, "loops": t.spec.loops,
            "mix_start": ms, "mix_end": me,
            "next_beats": kinds[i] if i < len(kinds) else None,  # 次の曲へのクロスフェード拍数（0 = エコーアウト）
        })
    report(1.0, "完了")
    return {"bpm": bpm, "duration": out.shape[1] / SR, "tracks": items}


def describe_transition(beats):
    if not beats:
        return "エコーアウトで切り替え"
    return (f"{beats // 4}小節" if beats % 4 == 0 else f"{beats}拍") + "のクロスフェード（ベーススワップ）"


def main():
    ap = argparse.ArgumentParser(description="聴かせたい範囲を指定して曲をつなぐ AI DJ 試作")
    ap.add_argument("tracks", nargs="+", help='"曲.mp3@1:05-2:10" の形式。末尾に x2 でループ')
    ap.add_argument("-o", "--output", default="mix.wav")
    ap.add_argument("--bars", type=int, default=8, help="つなぎの長さ（小節）。既定 8")
    ap.add_argument("--bpm", type=float, help="ミックス全体のテンポ。省略すると自動")
    ap.add_argument("--max-stretch", type=float, default=0.15,
                    help="テンポを変えてよい最大の割合。既定 0.15 (=±15%%)")
    args = ap.parse_args()

    try:
        result = make_mix([parse_spec(s) for s in args.tracks], args.output, args.bars, args.bpm,
                          args.max_stretch, progress=lambda frac, msg: print(msg))
    except MixError as e:
        sys.exit(str(e))

    bpm = result["bpm"]
    print(f"\nミックスのテンポ: {bpm:.1f} BPM" if bpm else "\nミックスのテンポ: なし（ビートのある曲がない）")
    for i, t in enumerate(result["tracks"]):
        tempo = f"{t['bpm']:.1f} BPM" if t["bpm"] else "テンポ不定"
        change = f" → {t['stretch'] * 100:+.1f}%" if t["stretch"] is not None else ("（テンポ差が大きいので変えない）" if t["bpm"] else "")
        loops = f" ×{t['loops']}" if t["loops"] > 1 else ""
        print(f"[{i + 1}] {t['path']}  {tempo}{change}")
        print(f"    範囲 {fmt(t['start'])}-{fmt(t['end'])}{loops}  →  ミックス内 {fmt(t['mix_start'])}-{fmt(t['mix_end'])}")
        if t["next_beats"] is not None:
            print(f"    ↓ {describe_transition(t['next_beats'])}")
    print(f"\n書き出し: {args.output}  ({fmt(result['duration'])})")


if __name__ == "__main__":
    main()
