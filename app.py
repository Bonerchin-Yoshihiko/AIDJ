#!/usr/bin/env python3
"""AI DJ デスクトップアプリ。

  python app.py            # アプリのウィンドウで開く
  python app.py --browser  # いつものブラウザで開く（ウィンドウが開けない環境向け）

画面（web/）は手元だけで動く小さなサーバーから配信し、それを pywebview のウィンドウで表示する。
"""
import argparse
import atexit
import mimetypes
import os
import shutil
import sys
import tempfile
import threading
import traceback
import uuid

FROZEN = getattr(sys, "frozen", False)
HERE = sys._MEIPASS if FROZEN else os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(sys.executable) if FROZEN else HERE

# ウィンドウだけのアプリとして固めると標準出力がないので、ログファイル（%TEMP%\aidj.log など）に書く
LOG_PATH = os.path.join(tempfile.gettempdir(), "aidj.log")
if sys.stdout is None or sys.stderr is None:
    _log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log
# librosa が使う numba のキャッシュを書き込める場所に置く
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "aidj-numba"))
# アプリと同じフォルダか rubberband/ フォルダに rubberband を置けば、高音質なテンポ調整が使える
os.environ["PATH"] = os.pathsep.join([APP_DIR, os.path.join(APP_DIR, "rubberband"), os.environ.get("PATH", "")])
# Windows ではレジストリの設定で .js が text/plain になり、画面が動かないことがある
mimetypes.add_type("text/javascript", ".js")

import soundfile as sf  # noqa: E402
from flask import Flask, abort, jsonify, request, send_file  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

import aidj  # noqa: E402

WORK = tempfile.mkdtemp(prefix="aidj-")
atexit.register(shutil.rmtree, WORK, ignore_errors=True)
# mp3 以外は、画面で必ず再生できる 16bit の wav をプレビュー用に作る
# （64bit 浮動小数点や ADPCM 圧縮の WAV は、画面のブラウザ部分では再生できないため）
PREVIEWABLE = {".mp3"}

app = Flask(__name__, static_folder=os.path.join(HERE, "web"), static_url_path="/static")
tracks = {}          # id -> {"path", "preview", "name"}
jobs = {}            # id -> {"state", "progress", "message", "result", "error", "path"}
mix_lock = threading.Lock()
window = None        # pywebview のウィンドウ（ブラウザで開いたときは None）


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/info")
def info():
    return jsonify(rubberband=shutil.which("rubberband") is not None, log=LOG_PATH)


@app.post("/api/tracks")
def add_track():
    f = request.files.get("file") or abort(400)
    tid = uuid.uuid4().hex[:12]
    ext = os.path.splitext(f.filename)[1].lower()
    path = os.path.join(WORK, tid + ext)
    f.save(path)
    try:
        y = aidj.load(path)
        info = aidj.overview(path, y)
        preview = path
        if ext not in PREVIEWABLE:
            preview = os.path.join(WORK, tid + "-preview.wav")
            sf.write(preview, y.T, aidj.SR, subtype="PCM_16")
    except Exception as e:
        os.remove(path)
        if isinstance(e, aidj.MixError):  # ファイル名は画面側で付けるので、理由だけ返す
            return jsonify(error=str(e).replace(f"{os.path.basename(path)}: ", "")), 400
        traceback.print_exc()  # 想定外のエラーはログに残す
        return jsonify(error=f"読み込み中にエラーが起きました（{e}）。詳しくは {LOG_PATH}"), 500
    tracks[tid] = {"path": path, "preview": preview, "name": f.filename}
    return jsonify(id=tid, name=f.filename, **info)


@app.get("/api/tracks/<tid>/audio")
def track_audio(tid):
    t = tracks.get(tid) or abort(404)
    return send_file(t["preview"], conditional=True)


@app.post("/api/mix")
def start_mix():
    body = request.get_json(force=True)
    items = body.get("items") or []
    if not items:
        return jsonify(error="曲を追加してください"), 400
    specs = []
    for it in items:
        t = tracks.get(it["id"]) or abort(400)
        specs.append(aidj.Spec(t["path"], float(it["start"]), float(it["end"]), int(it.get("loops", 1)), t["name"]))

    jid = uuid.uuid4().hex[:12]
    job = jobs[jid] = {"state": "running", "progress": 0.0, "message": "順番待ち",
                       "path": os.path.join(WORK, f"mix-{jid}.wav")}

    def run():
        try:
            with mix_lock:
                res = aidj.make_mix(specs, job["path"], bars=int(body.get("bars", 8)),
                                    bpm=float(body["bpm"]) if body.get("bpm") else None,
                                    max_stretch=float(body.get("max_stretch", 0.15)),
                                    bridges=body.get("bridge", "auto"),
                                    progress=lambda frac, msg: job.update(progress=frac, message=msg))
            for t, it in zip(res["tracks"], items):
                t.pop("path")
                t["id"] = it["id"]
            y, _ = sf.read(job["path"], dtype="float32")
            res["peaks"] = aidj.peaks_of(y.mean(axis=1))
            job.update(state="done", result=res)
        except aidj.MixError as e:
            job.update(state="error", error=str(e))
        except Exception as e:  # 想定外のエラーも画面に出す
            traceback.print_exc()
            job.update(state="error", error=f"ミックス中にエラーが起きました: {e}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify(id=jid)


@app.get("/api/mix/<jid>")
def mix_status(jid):
    job = jobs.get(jid) or abort(404)
    return jsonify({k: v for k, v in job.items() if k != "path"})


@app.get("/api/mix/<jid>/audio")
def mix_audio(jid):
    job = jobs.get(jid) or abort(404)
    if job["state"] != "done":
        abort(404)
    return send_file(job["path"], conditional=True,
                     as_attachment="download" in request.args, download_name="aidj-mix.wav")


@app.post("/api/mix/<jid>/save")
def save_mix(jid):
    """アプリのウィンドウなら保存ダイアログを出す。ブラウザなら画面側でダウンロードさせる。"""
    job = jobs.get(jid) or abort(404)
    if window is None:
        return jsonify(dialog=False)
    import webview
    dest = window.create_file_dialog(webview.FileDialog.SAVE, save_filename="aidj-mix.wav",
                                     file_types=("WAV (*.wav)", "MP3 (*.mp3)"))
    if not dest:
        return jsonify(saved=None)
    dest = dest if isinstance(dest, str) else dest[0]
    if dest.lower().endswith(".mp3"):
        y, sr = sf.read(job["path"])
        sf.write(dest, y, sr)
    else:
        shutil.copyfile(job["path"], dest)
    return jsonify(saved=dest)


def main():
    global window
    ap = argparse.ArgumentParser(description="AI DJ デスクトップアプリ")
    ap.add_argument("--browser", action="store_true", help="アプリのウィンドウではなくブラウザで開く")
    ap.add_argument("--no-open", action="store_true", help="サーバーだけ起動する（開発・テスト用）")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    server = make_server("127.0.0.1", args.port, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"

    if not (args.browser or args.no_open):
        try:
            import webview
            window = webview.create_window("AI DJ", url, width=1200, height=860, min_size=(860, 600),
                                           background_color="#0e1014")
            webview.start()
            return
        except Exception as e:
            print(f"アプリのウィンドウを開けなかったので、ブラウザで開きます（{e}）")
            window = None
    if not args.no_open:
        import webbrowser
        webbrowser.open(url)
    print(f"AI DJ: {url} で動いています。終了は Ctrl+C", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
