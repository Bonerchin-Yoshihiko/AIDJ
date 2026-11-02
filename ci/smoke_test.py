#!/usr/bin/env python3
"""CI 用: 作った exe を起動して、曲の読み込み → ミックス → 書き出しまで通るか確かめる。

  python ci/smoke_test.py "dist/AI DJ/AI DJ.exe"
"""
import os
import subprocess
import sys
import tempfile
import time

import requests
import soundfile as sf

URL = "http://127.0.0.1:8765"
LOG = os.path.join(tempfile.gettempdir(), "aidj.log")
OUT = "ci-output"


def wait_for_server(proc):
    for _ in range(180):
        if proc.poll() is not None:
            sys.exit(f"アプリが終了してしまいました（終了コード {proc.returncode}）")
        try:
            if requests.get(URL, timeout=2).ok:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    sys.exit("アプリのサーバーが起動しませんでした")


def upload(path, name):
    with open(path, "rb") as f:
        r = requests.post(f"{URL}/api/tracks", files={"file": (name, f)}, timeout=600)
    if not r.ok:
        sys.exit(f"{name} の読み込みに失敗: {r.status_code} {r.text}")
    info = r.json()
    print(f"読み込み OK: {info['name']}  BPM={info['bpm']}  テンポ一定={info['steady']}  おすすめ範囲={info['suggested']}")
    return info


def wav_variants():
    """いろいろな種類の WAV を作る（48kHz/24bit、64bit 浮動小数点、ADPCM 圧縮、ヘッダーが壊れたもの など）。"""
    y, sr = sf.read("demo/house_128.wav", frames=44100 * 20)
    d = os.path.join(OUT, "wavs")
    os.makedirs(d, exist_ok=True)
    files = {
        "48k_24bit.wav": dict(samplerate=48000, subtype="PCM_24"),
        "float64.wav": dict(subtype="DOUBLE"),
        "ms_adpcm.wav": dict(subtype="MS_ADPCM"),
        "ima_adpcm.wav": dict(subtype="IMA_ADPCM"),
        "8bit_mono.wav": dict(subtype="PCM_U8", mono=True),
        "extensible.wav": dict(subtype="PCM_16", format="WAVEX"),
        "header_broken.wav": dict(subtype="PCM_16"),
    }
    durations = {}
    for name, opt in files.items():
        data = y.mean(axis=1) if opt.pop("mono", False) else y
        rate = opt.pop("samplerate", sr)
        sf.write(os.path.join(d, name), data, rate, **opt)
        durations[name] = len(data) / rate
    with open(os.path.join(d, "header_broken.wav"), "r+b") as f:  # data チャンクの大きさを 0 にする
        raw = f.read()
        f.seek(raw.find(b"data") + 4)
        f.write(b"\0\0\0\0")
    with open(os.path.join(d, "actually_m4a.wav"), "wb") as f:  # 中身が m4a の「.wav」
        f.write(b"\0\0\0\x20ftypM4A " + bytes(2000))
    return d, durations


def check_wavs():
    d, durations = wav_variants()
    for name, duration in durations.items():
        info = upload(os.path.join(d, name), name)
        assert abs(info["duration"] - duration) < 0.1, f"{name} の長さがおかしい: {info['duration']}"
        with open(os.path.join(d, "preview.wav"), "wb") as f:
            f.write(requests.get(f"{URL}/api/tracks/{info['id']}/audio").content)
        p = sf.info(os.path.join(d, "preview.wav"))
        assert p.subtype == "PCM_16", f"{name} のプレビューが再生しやすい形式になっていない: {p.subtype}"
    r = requests.post(f"{URL}/api/tracks", files={"file": ("actually_m4a.wav", open(os.path.join(d, "actually_m4a.wav"), "rb"))})
    assert r.status_code == 400 and "m4a" in r.json()["error"], f"中身が m4a のときの説明がおかしい: {r.text}"
    print("いろいろな WAV: すべて OK /", r.json()["error"])


def main():
    os.makedirs(OUT, exist_ok=True)
    # mp3 の読み込みと日本語のファイル名も確かめる
    y, sr = sf.read("demo/house_124.wav")
    sf.write(os.path.join(OUT, "house_124.mp3"), y, sr)

    proc = subprocess.Popen([sys.argv[1], "--no-open", "--port", "8765"])
    try:
        wait_for_server(proc)
        print("サーバー起動 OK", requests.get(f"{URL}/api/info").json())
        a = upload(os.path.join(OUT, "house_124.mp3"), "テスト曲A.mp3")
        b = upload("demo/house_128.wav", "house_128.wav")
        c = upload("demo/piano_free.wav", "piano_free.wav")
        assert a["name"] == "テスト曲A.mp3"
        assert abs(a["bpm"] - 124) < 1 and abs(b["bpm"] - 128) < 1 and not c["steady"], "BPM の検出がおかしい"
        check_wavs()

        items = [{"id": a["id"], "start": 31, "end": 62}, {"id": b["id"], "start": 30, "end": 45, "loops": 2},
                 {"id": c["id"], "start": 10, "end": 40}]
        jid = requests.post(f"{URL}/api/mix", json={"items": items, "bars": 8}).json()["id"]
        t0 = time.time()
        while (job := requests.get(f"{URL}/api/mix/{jid}").json())["state"] == "running":
            if time.time() - t0 > 900:
                sys.exit("ミックスが終わりません")
            time.sleep(1)
        if job["state"] != "done":
            sys.exit(f"ミックスに失敗: {job.get('error')}")
        res = job["result"]
        print(f"ミックス OK（{time.time() - t0:.0f}秒）: {res['bpm']:.1f} BPM, {res['duration']:.1f}秒")
        for t in res["tracks"]:
            print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})

        mix_path = os.path.join(OUT, "mix.wav")
        with open(mix_path, "wb") as f:
            f.write(requests.get(f"{URL}/api/mix/{jid}/audio").content)
        y, sr = sf.read(mix_path)
        sf.write(os.path.join(OUT, "mix.mp3"), y, sr)
        os.remove(mix_path)

        assert abs(res["bpm"] - 126) < 0.5, "ミックスのテンポがおかしい"
        assert res["tracks"][0]["next_beats"] == 32, "1→2曲目が8小節のクロスフェードになっていない"
        assert res["tracks"][1]["next_beats"] == 0, "2→3曲目がエコーアウトになっていない"
        assert 110 < len(y) / sr < 130, "ミックスの長さがおかしい"
        print("すべて OK")
    finally:
        proc.kill()
        if os.path.exists(LOG):
            print("---- アプリのログ（最後の部分） ----")
            with open(LOG, encoding="utf-8", errors="replace") as f:
                print(f.read()[-4000:])


if __name__ == "__main__":
    main()
