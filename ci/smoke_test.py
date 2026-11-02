#!/usr/bin/env python3
"""CI 用: 作った exe を起動して、曲の読み込み → ミックス → 書き出しまで通るか確かめる。

  python ci/smoke_test.py "dist/Miso Soup/Miso Soup.exe"
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import requests
import soundfile as sf

URL = "http://127.0.0.1:8765"
LOG = os.path.join(tempfile.gettempdir(), "miso-soup.log")
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


def run_mix(items, bridge, save_as, links=None):
    body = {"items": items, "bars": 8, "bridge": bridge, "links": links}
    jid = requests.post(f"{URL}/api/mix", json=body).json()["id"]
    t0 = time.time()
    while (job := requests.get(f"{URL}/api/mix/{jid}").json())["state"] == "running":
        if time.time() - t0 > 900:
            sys.exit("ミックスが終わりません")
        time.sleep(1)
    if job["state"] != "done":
        sys.exit(f"ミックスに失敗: {job.get('error')}")
    res = job["result"]
    print(f"ミックス OK（ブリッジ={bridge}, {time.time() - t0:.0f}秒）: {res['bpm']:.1f} BPM, {res['duration']:.1f}秒")
    for t in res["tracks"]:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})
    mix_path = os.path.join(OUT, "mix.wav")
    with open(mix_path, "wb") as f:
        f.write(requests.get(f"{URL}/api/mix/{jid}/audio").content)
    y, sr = sf.read(mix_path)
    sf.write(os.path.join(OUT, save_as), y, sr)
    os.remove(mix_path)
    return res, y, sr


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

        # ブリッジなし: 1→2 は 8小節のクロスフェード、2→3（テンポ不定のピアノ）はエコーアウト
        items = [{"id": a["id"], "start": 31, "end": 62}, {"id": b["id"], "start": 30, "end": 45, "loops": 2},
                 {"id": c["id"], "start": 10, "end": 40}]
        res, y, sr = run_mix(items, "off", "mix_no_bridge.mp3")
        assert abs(res["bpm"] - 126) < 0.5, "ミックスのテンポがおかしい"
        assert res["tracks"][0]["next"] == {"kind": "crossfade", "beats": 32}, "1→2曲目が8小節のクロスフェードになっていない"
        assert res["tracks"][1]["next"] == {"kind": "echo"}, "2→3曲目がエコーアウトになっていない"
        assert 110 < len(y) / sr < 130, "ミックスの長さがおかしい"

        # ブリッジあり: テンポもキーも違う 90 BPM・F# メジャーの曲と、テンポ不定のピアノの前にブリッジが入る
        g = upload("demo/groove_90.wav", "groove_90.wav")
        assert g["key"] == "F# メジャー", f"キーの推定がおかしい: {g['key']}"
        items = [{"id": a["id"], "start": 31, "end": 62}, {"id": b["id"], "start": 30, "end": 45},
                 {"id": g["id"], "start": 42.7, "end": 85.3}, {"id": c["id"], "start": 10, "end": 40}]
        res, y, sr = run_mix(items, "auto", "mix_bridge.mp3")
        kinds = [t["next"]["kind"] for t in res["tracks"][:-1]]
        assert kinds == ["crossfade", "bridge", "bridge"], f"つなぎ方がおかしい: {kinds}"
        br = res["tracks"][1]["next"]
        assert abs(br["from_bpm"] - res["bpm"]) < 1 and abs(br["to_bpm"] - 90) < 1, f"ブリッジのテンポがおかしい: {br}"
        assert br["to_key"] == "F# メジャー", f"ブリッジの行き先のキーがおかしい: {br}"
        assert res["tracks"][2]["next"]["to_bpm"] is None, "テンポ不定の曲へのブリッジがおかしい"

        # 曲のパートを使ったブリッジ（AI のパート分け）
        info = requests.get(f"{URL}/api/info").json()
        assert info["stems"], "AI のパート分けが使えない"
        items = [{"id": b["id"], "start": 30, "end": 45}, {"id": g["id"], "start": 42.7, "end": 85.3}]
        link = {"a": ["drums", "vocals"], "b": ["drums", "other"], "synth": False}
        res, y, sr = run_mix(items, "auto", "mix_stems.mp3", links=[link])
        br = res["tracks"][0]["next"]
        assert br["kind"] == "bridge", f"パートを選んだのにブリッジになっていない: {br}"
        # ボーカルは、AI が声と認識できなければ「見つからなかった」として外される（その場合も正しい動き）
        vocals_ok = br["parts"]["a"] == ["drums", "vocals"] and not br["missing"]
        vocals_missing = br["parts"]["a"] == ["drums"] and br["missing"] == ["前の曲のボーカル"]
        assert (vocals_ok or vocals_missing) and br["parts"]["b"] == ["drums", "other"], f"使ったパートがおかしい: {br}"
        assert not br["synth"], f"合成音を切ったのに使われている: {br}"
        print("ボーカルチョップ:", "使えた" if vocals_ok else "声が見つからず外れた（テスト用の声が声と認識されなかった）")
        assert np.sqrt(np.mean(y ** 2)) > 0.01, "ミックスがほぼ無音"
        print("すべて OK")
    finally:
        proc.kill()
        if os.path.exists(LOG):
            print("---- アプリのログ（最後の部分） ----")
            with open(LOG, encoding="utf-8", errors="replace") as f:
                print(f.read()[-4000:])


if __name__ == "__main__":
    main()
