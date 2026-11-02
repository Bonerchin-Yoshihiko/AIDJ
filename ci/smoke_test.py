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
