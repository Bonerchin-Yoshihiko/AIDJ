"""テンポやキーが違う曲の間に入れる「つなぎのビート（ブリッジ）」を合成する。

前の曲のテンポとキーから始まり、テンポを少しずつ変えながら転調して、
次の曲のテンポで、次の曲の属和音（次の曲のキーに戻りたくなる和音）で終わる。
最初と最後の数小節はテンポを固定してあるので、前後の曲とビートを合わせて重ねられる。
"""
import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100
NOTE_NAMES = ["C", "C#", "D", "E♭", "E", "F", "F#", "G", "A♭", "A", "B♭", "B"]
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
# Krumhansl のキー・プロファイル（そのキーで各音がどれくらい使われるか）
PROFILES = {
    "major": np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "minor": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
}


# ---------- キー

def estimate_key(chroma):
    """12音の強さ（クロマ）から、一番それらしいキー (主音 0〜11, "major"/"minor") を選ぶ。"""
    chroma = np.asarray(chroma, float)
    scores = {(tonic, mode): np.corrcoef(chroma, np.roll(prof, tonic))[0, 1]
              for mode, prof in PROFILES.items() for tonic in range(12)}
    return max(scores, key=scores.get)


def musical_tempo(bpm):
    """倍・半分にして 80〜160 BPM に収める（ビート検出が倍のテンポを拾ったときなど）。"""
    while bpm >= 160:
        bpm /= 2
    while bpm < 80:
        bpm *= 2
    return bpm


def key_name(key):
    return f"{NOTE_NAMES[key[0]]} {'メジャー' if key[1] == 'major' else 'マイナー'}"


def key_distance(a, b):
    """五度圏での距離（0〜6）。平行調（C メジャーと A マイナーなど）は 0。"""
    fa, fb = [((k[0] + (3 if k[1] == "minor" else 0)) * 7) % 12 for k in (a, b)]
    d = (fa - fb) % 12
    return min(d, 12 - d)


def diatonic(key):
    """そのキーで使える三和音（減三和音は除く）。"""
    tonic, mode = key
    s = SCALES[mode]
    chords = []
    for i in range(7):
        third, fifth = (s[(i + 2) % 7] - s[i]) % 12, (s[(i + 4) % 7] - s[i]) % 12
        if fifth == 7:
            chords.append(((tonic + s[i]) % 12, "major" if third == 4 else "minor"))
    return chords


def chord_path(a, b):
    """キー a からキー b へ移る4つのコード。

    a の主和音 → 両方のキーに共通するコード（なければ a の下属和音）→ b の下属和音 → b の属和音
    """
    home = (a[0], a[1])
    dominant = ((b[0] + 7) % 12, "major")  # 短調でも属和音は長三和音にする（和声的短音階）
    if b[1] == "major":
        pre_b = [((b[0] + 2) % 12, "minor"), ((b[0] + 5) % 12, "major")]  # ii, IV
    else:
        pre_b = [((b[0] + 5) % 12, "minor"), ((b[0] + 8) % 12, "major")]  # iv, VI
    common = [c for c in diatonic(a) if c in diatonic(b) and c not in (home, dominant)]
    pivot = next((c for c in pre_b + common if c in common), ((a[0] + 5) % 12, a[1]))
    pre = next(c for c in pre_b if c != pivot)
    return [home, pivot, pre, dominant]


def voicing(chord, low=55):
    """コードの3音を low〜low+12 の範囲に並べる（近い音同士でつながるように）。"""
    root, quality = chord
    return sorted(low + (root + t - low) % 12 for t in (0, 4 if quality == "major" else 3, 7))


# ---------- 音づくり

rng = np.random.default_rng(7)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def lowpass(f, fc):
    """周波数 f の倍音が、カットオフ fc のローパスでどれくらい残るか。"""
    return 1 / np.sqrt(1 + (f / fc) ** 8)


def add(out, x, at):
    i = int(round(at * SR))
    if x.ndim == 1:
        x = np.stack([x, x])
    j = min(out.shape[1], i + x.shape[1])
    if 0 <= i < out.shape[1]:
        out[:, i:j] += x[:, :j - i]


def decay(n, sec):
    return np.exp(-np.arange(n) / (sec * SR))


def kick():
    n = int(0.4 * SR)
    t = np.arange(n) / SR
    freq = 48 + 110 * np.exp(-t / 0.025)
    click = np.concatenate([rng.standard_normal(int(0.003 * SR)) * 0.3, np.zeros(n - int(0.003 * SR))])
    return (np.sin(2 * np.pi * np.cumsum(freq) / SR) + click) * decay(n, 0.16)


def noise_hit(sec, band):
    n = int(sec * 6 * SR)
    sos = butter(2, band, btype="bandpass" if isinstance(band, list) else "highpass", fs=SR, output="sos")
    return sosfilt(sos, rng.standard_normal(n)) * decay(n, sec)


def clap():
    x = np.zeros(int(0.3 * SR))
    for k, d in enumerate((0, 0.011, 0.022)):  # 手拍子っぽく3回重ねる
        hit = noise_hit(0.03 if k < 2 else 0.12, [900, 4000])
        i = int(d * SR)
        x[i:i + len(hit)] += hit[:len(x) - i]
    return x


def riser(sec):
    """だんだん高くなるノイズ（盛り上げ用）。"""
    n = int(sec * SR)
    noise = rng.standard_normal(n)
    t = np.linspace(0, 1, n)
    bands = [300, 600, 1200, 2400, 4800, 9000]
    out = np.zeros(n)
    for i, fc in enumerate(bands):
        sos = butter(2, [fc / 1.6, min(fc * 1.6, 20000)], btype="bandpass", fs=SR, output="sos")
        out += sosfilt(sos, noise) * np.exp(-((t * (len(bands) - 1) - i) ** 2) / 0.7)
    return out * t ** 2


def pad(out, start, end, notes, cutoff, level):
    """デチューンした2つの声部（左右）で鳴らすパッド。cutoff はブリッジ全体の時間ごとの値。"""
    rel = 0.4
    i0, i1 = int(start * SR), min(out.shape[1], int((end + rel) * SR))
    t = np.arange(i1 - i0) / SR
    env = np.minimum(1, t / 0.12) * np.clip((end - start + rel - t) / rel, 0, 1)
    fc = cutoff[i0:i1]
    for m in notes:
        for ch, cents in ((0, -7), (1, 7)):
            f = hz(m) * 2 ** (cents / 1200)
            for h in range(1, 16):
                if f * h > 14000:
                    break
                out[ch, i0:i1] += level * env * lowpass(f * h, fc) / h * np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6.3))


def pluck(midi, sec, fc, pan):
    n = int(sec * SR)
    t = np.arange(n) / SR
    f = hz(midi)
    x = sum(lowpass(f * h, fc) / h * np.sin(2 * np.pi * f * h * t) for h in range(1, 10) if f * h < 14000)
    x = x * np.minimum(1, t / 0.003) * decay(n, 0.14)
    return np.stack([x * (1 - pan), x * (1 + pan)])


def bass(midi, sec):
    n = int(sec * SR)
    t = np.arange(n) / SR
    f = hz(midi)
    x = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t) + 0.15 * np.sin(6 * np.pi * f * t)
    env = np.minimum(1, t / 0.005) * decay(n, sec * 0.6) * np.clip((sec - t) / 0.02, 0, 1)
    return np.tanh(1.5 * x) * env


# ---------- ブリッジ本体

def beat_grid(t_a, t_b, beats, hold):
    """各拍の開始時刻（最後に終わりの時刻も入る）。最初と最後の hold 拍はテンポ固定で、その間は滑らかに変える。"""
    times = [0.0]
    for k in range(beats):
        x = min(1.0, max(0.0, (k + 0.5 - hold) / max(1, beats - 2 * hold)))
        times.append(times[-1] + 60 / (t_a * (t_b / t_a) ** (x * x * (3 - 2 * x))))
    return np.array(times)


def make_bridge(t_a, t_b, key_a, key_b, bars):
    """ブリッジを合成する。t_a / t_b は前後の曲のテンポ（None = テンポ不定でビートなし）。

    戻り値: (音声 (2, n), 前後それぞれのテンポ固定部分の拍数)
    """
    beats, hold = 4 * bars, 4 * max(1, bars // 4)
    drums_in, drums_out = t_a is not None, t_b is not None
    t_a = t_a or t_b or 100.0
    t_b = t_b or t_a
    times = beat_grid(t_a, t_b, beats, hold)
    length = int(round(times[-1] * SR))
    out = np.zeros((2, length + SR))
    sub = lambda k, s: times[k] + (times[k + 1] - times[k]) * s  # k 拍目の中の位置 s (0〜1)

    # コード: 4つのコードをブリッジ全体に均等に割り当てる
    chords = chord_path(key_a, key_b)
    chord_at = lambda k: chords[min(3, k * 4 // beats)]
    cutoff = 700 * (5000 / 700) ** np.linspace(0, 1, out.shape[1])  # パッドの音色をだんだん明るく
    for i, c in enumerate(chords):
        k0, k1 = i * beats // 4, (i + 1) * beats // 4
        pad(out, times[k0], times[k1], voicing(c), cutoff, 0.05)

    k_kick, k_clap, k_hat = kick(), clap(), noise_hit(0.02, 7000)
    for k in range(beats):
        last_bar = k >= beats - 4
        drums = (k >= hold or drums_in) and (k < beats - hold or drums_out)
        root = chord_at(k)[0]
        if drums and not (last_bar and drums_out):
            add(out, 0.9 * k_kick, times[k])
            if k % 4 in (1, 3):
                add(out, 0.35 * k_clap, times[k])
            add(out, 0.12 * k_hat, sub(k, 0.5))
            add(out, 0.3 * bass(33 + (root - 33) % 12, (times[k + 1] - times[k]) * 0.45), sub(k, 0.5))
            if hold <= k < beats - hold:  # テンポが変わっている間は16分のハイハットで勢いをつける
                for s in (0.25, 0.75):
                    add(out, 0.05 * k_hat, sub(k, s))
        if hold <= k < beats - 4:  # アルペジオ（コードの音を16分で）を少しずつ大きく
            tones = voicing(chord_at(k), 67) + [voicing(chord_at(k), 67)[0] + 12]
            vol = 0.03 + 0.05 * (k - hold) / max(1, beats - 4 - hold)
            for s in range(4):
                fc = cutoff[int(sub(k, s / 4) * SR)]
                add(out, vol * pluck(tones[(k * 4 + s) % 4], 0.3, fc, 0.4 if s % 2 else -0.4), sub(k, s / 4))
        if last_bar and drums_out:  # 最後の1小節はスネアロールで次の曲へ
            steps = 4 if k < beats - 2 else 8
            for s in range(steps):
                pos = (k - (beats - 4) + s / steps) / 4
                add(out, (0.1 + 0.5 * pos) * k_clap, sub(k, s / steps))
    if drums_out:
        r = riser(times[-1] - times[beats - hold])
        add(out, 0.12 * r, times[-1] - len(r) / SR)

    out = out[:, :length]
    fade = int(0.01 * SR)
    out[:, -fade:] *= np.linspace(1, 0, fade)
    return out.astype(np.float32), hold
